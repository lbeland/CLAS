// ---------------------------------------------------------------------
// This file is part of falcon-core.
//
// Copyright (C) 2015, 2016, 2017 Neuro-Electronics Research Flanders
//
// Falcon-server is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Falcon-server is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with falcon-core. If not, see <http://www.gnu.org/licenses/>.
// ---------------------------------------------------------------------

#include "IAFEstimator.hpp"
#include "utilities/time.hpp"
#include "logging/log.hpp"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <limits>
#include <sstream>
#include <string>
#include <fftw3.h>
#include <complex>
#include <dsp/fftw_planner_mutex.hpp>


IAFEstimator::IAFEstimator() : IProcessor(PRIORITY_HIGH) {
  add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
  add_option("n_fft", n_fft_, "FFT size");

  iaf_state_ = create_broadcaster_state<float>(
      "iaf", current_iaf_, Permission::NONE,
      "Individual alpha frequency shared with downstream processors.");
}

void IAFEstimator::CreatePorts() {
  data_in_port_ = create_input_port<MultiChannelType<float>>(
      "in", 
      MultiChannelType<float>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
      PortInPolicy(SlotRange(0,MAX_NCHANNELS)));
}

void IAFEstimator::Prepare(GlobalContext &context) {
  const auto& info = data_in_port_->streaminfo(0);
  const auto& p = info.parameters<MultiChannelType<float>::Parameters>();
  LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate << "\n";
  fs_ = p.sample_rate;
  int N = static_cast<int>(fs_ * 1);  // 1 second
  sample_window.set_capacity(N);
  LOG(INFO) << name() << " Sample window size set to " << sample_window.capacity() << "\n";

  const int n_fft = static_cast<int>(n_fft_());
  if (n_fft < N) {
    throw std::runtime_error("IAFEstimator: n_fft must be >= window length");
  }

  iaf_state_->set(current_iaf_);
}

int IAFEstimator::get_max_bin(fftwf_complex* freq, size_t N) {
  int max_bin = -1;
  double max_mag2 = -1.0;

  for (size_t k = 0; k < N; ++k) {
    double mag2 = pow(freq[k][0],2) + pow(freq[k][1],2);

    if (mag2 > max_mag2) {
      max_mag2 = mag2;
      max_bin = static_cast<int>(k);
    }
  }
  return max_bin;
}


void IAFEstimator::fftshift(const fftwf_complex* in, fftwf_complex* out, int L) {
    int s = L / 2;  // floor(L/2)
    for (int k = 0; k < L; ++k) {
        int src = (k + s) % L;
        out[k][0] = in[src][0];
        out[k][1] = in[src][1];
    }
}

void IAFEstimator::ifftshift(const fftwf_complex* in, fftwf_complex* out, int L) {
    int s = (L + 1) / 2;  // ceil(L/2)
    for (int k = 0; k < L; ++k) {
        int src = (k + s) % L;
        out[k][0] = in[src][0];
        out[k][1] = in[src][1];
    }
}

void IAFEstimator::ifftshift(const std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out, int L)
{
    int s = (L + 1) / 2;  // ceil(L/2)
    for (int k = 0; k < L; ++k) {
        int src = (k + s) % L;
        out[k] = in[src];
    }
}

void IAFEstimator::Process(ProcessingContext &context) {
  std::vector<MultiChannelType<float>::Data*> data_in;

  // FFTW output spectrum
  const int N = static_cast<int>(sample_window.capacity());
  const int n_fft = n_fft_();  // FFT size
  float* signal_in = fftwf_alloc_real(n_fft);
  fftwf_complex* freq_half = fftwf_alloc_complex(n_fft/2 + 1);
  fftwf_complex* freq = fftwf_alloc_complex(n_fft);

  fftwf_plan p;
  {
    std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
    p = fftwf_plan_dft_r2c_1d(n_fft, signal_in, freq_half, FFTW_ESTIMATE);
  }

  // Measurement phase
  while (!context.terminated()) {

    if (n_messages_() != -1 && packet_count_ >= n_messages_()) {
      break;
    }
  
    // Try to retrieve data
    if (!data_in_port_->slot(0)->RetrieveDataAll(data_in)) {
      break;
    }

    for (const auto* packet : data_in) {
      for (size_t s = 0; s < packet->nsamples(); ++s) {
        sample_window.push_back(packet->data_sample(s, 0));
        packet_count_++;
      }
    }

    data_in_port_->slot(0)->ReleaseData();

    if ((sample_window.size() == sample_window.capacity())) { //} && (packet_count_ % (sample_window.capacity()/2) == 0)) {

      // Convert circular buffer<float> to continuous array for FFTW input and zero-pad to n_fft
      // signal_in = sample_window.linearize();
      for (int i = 0; i < N; ++i) {
          signal_in[i] = sample_window[i];
      }
      for (int i = N; i < n_fft; ++i) {
          signal_in[i] = 0.0f;
      }

      // FFT
      fftwf_execute(p);

      // Get max bin and convert to frequency
      int max_bin = get_max_bin(freq_half, n_fft/2 + 1);
      float freq_resolution = static_cast<float>(fs_) / n_fft;
      if (std::abs(max_bin * freq_resolution - current_iaf_) > 1e-3f) {
        current_iaf_ = max_bin * freq_resolution;
        printf("\n Packet %d: Estimated IAF = %.2f Hz (max bin: %d)", packet_count_, current_iaf_, max_bin);
        
        iaf_state_->set(current_iaf_);  
      // } else {
      //   printf("\n IAF Estimation did not change: %.2fHz", current_iaf_);
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
    fftwf_destroy_plan(p);
  }
  fftwf_free(signal_in);
  fftwf_free(freq_half);
  fftwf_free(freq);
}

void IAFEstimator::Postprocess(ProcessingContext &context) {
  printf("\n ---------------- \n IAFEstimator: Total messages processed: %d", packet_count_);
}

REGISTERPROCESSOR(IAFEstimator);
