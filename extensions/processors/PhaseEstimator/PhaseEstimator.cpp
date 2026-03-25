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

#include "PhaseEstimator.hpp"
#include "utilities/time.hpp"
#include "logging/log.hpp"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <limits>
#include <fftw3.h>
#include <complex>


PhaseEstimator::PhaseEstimator() : IProcessor(PRIORITY_HIGH) {
  add_option("n_messages", n_messages_, "Number of packets to receive");
  add_option("n_fft", n_fft_, "FFT size");
}

void PhaseEstimator::CreatePorts() {
  data_in_port_ = create_input_port<MultiChannelType<float>>(
      "in", 
      MultiChannelType<float>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
      PortInPolicy(SlotRange(1)));

  data_out_port_ = create_output_port<MultiChannelType<float>>(
      "out",
      MultiChannelType<float>::Parameters(1,1,1), // Placeholder, will be set in CompleteStreamInfo
      PortOutPolicy(SlotRange(2),200,WaitStrategy::kBlockingStrategy));
}

void PhaseEstimator::CompleteStreamInfo() {
  const auto &input_info = data_in_port_->slot(0)->streaminfo();
  const auto &input_params =
      input_info.parameters<MultiChannelType<float>::Parameters>();

  dynamic_cast<StreamInfo<MultiChannelType<float>>&>(
      data_out_port_->slot(0)->streaminfo())
      .set_parameters(input_params);

  dynamic_cast<StreamInfo<MultiChannelType<float>>&>(
      data_out_port_->slot(1)->streaminfo())
      .set_parameters(input_params);
}

void PhaseEstimator::Preprocess(ProcessingContext &context) {
  printf("\n");
  const auto& info = data_in_port_->streaminfo(0);
  const auto& p = info.parameters<MultiChannelType<float>::Parameters>();
  printf("Stream parameters - nchannels: %lu, nsamples: %lu, sample_rate: %f\n", p.nchannels, p.nsamples, p.sample_rate);
  fs_ = p.sample_rate;
  f0_ = 10.0;
  int N = static_cast<int>(fs_ * (1.0/f0_)*2.0);  // 2 cycles of a 10 Hz sine wave
  sample_window.set_capacity(N);
  printf("Sample window size set to %lu\n", sample_window.capacity());

  const int n_fft = static_cast<int>(n_fft_());
  if (n_fft < N) {
    throw std::runtime_error("PhaseEstimator: n_fft must be >= window length");
  }
}

int PhaseEstimator::get_max_bin(fftwf_complex* out, size_t out_size) {
  int max_bin = -1;
  double max_mag2 = -1.0;

  for (size_t k = 0; k < out_size; ++k) {
    double re = out[k][0];
    double im = out[k][1];
    double mag2 = re * re + im * im;

    if (mag2 > max_mag2) {
      max_mag2 = mag2;
      max_bin = static_cast<int>(k);
    }
  }
  return max_bin;
}


void PhaseEstimator::construct_analytic_spectrum(int n_fft, const fftwf_complex* half, fftwf_complex* full){
  for (int k = 0; k < n_fft; ++k) {
      full[k][0] = 0.0;
      full[k][1] = 0.0;
  }

  full[0][0] = half[0][0];
  full[0][1] = half[0][1];

  if (n_fft % 2 == 0) {
      for (int k = 1; k < n_fft / 2; ++k) {
          full[k][0] = 2.0 * half[k][0];
          full[k][1] = 2.0 * half[k][1];
      }

      full[n_fft / 2][0] = half[n_fft / 2][0];
      full[n_fft / 2][1] = half[n_fft / 2][1];
  } else {
      for (int k = 1; k <= (n_fft - 1) / 2; ++k) {
          full[k][0] = 2.0 * half[k][0];
          full[k][1] = 2.0 * half[k][1];
      }
  }
}

void PhaseEstimator::fftshift(const fftwf_complex* in, fftwf_complex* out, int L) {
    int s = L / 2;  // floor(L/2)
    for (int k = 0; k < L; ++k) {
        int src = (k + s) % L;
        out[k][0] = in[src][0];
        out[k][1] = in[src][1];
    }
}
void PhaseEstimator::ifftshift(const fftwf_complex* in, fftwf_complex* out, int L) {
    int s = (L + 1) / 2;  // ceil(L/2)
    for (int k = 0; k < L; ++k) {
        int src = (k + s) % L;
        out[k][0] = in[src][0];
        out[k][1] = in[src][1];
    }
}

void PhaseEstimator::Process(ProcessingContext &context) {
  MultiChannelType<float>::Data* data_in;
  MultiChannelType<float>::Data *data_real_out = nullptr;
  MultiChannelType<float>::Data *data_phase_out = nullptr;

  // FFTW output spectrum
  float sample;
  float phase;
  float real_part;
  const int N = static_cast<int>(sample_window.capacity());
  const int n_fft = n_fft_();  // FFT size
  float* signal_in = fftwf_alloc_real(n_fft);
  fftwf_complex* freq_half = fftwf_alloc_complex(n_fft/2 + 1);
  fftwf_complex* freq = fftwf_alloc_complex(n_fft);
  fftwf_complex* out = fftwf_alloc_complex(n_fft);

  fftwf_plan p = fftwf_plan_dft_r2c_1d(n_fft, signal_in, freq_half, FFTW_ESTIMATE);
  fftwf_plan p_inv = fftwf_plan_dft_1d(n_fft, freq, out, FFTW_BACKWARD, FFTW_ESTIMATE);

  // Measurement phase
  while (!context.terminated()) {
    if (n_messages_() != -1 && packet_count_ >= n_messages_()) {
      break;
    }
  
    // Try to retrieve data
    if (!data_in_port_->slot(0)->RetrieveData(data_in)) {
      break;
    }

    sample = data_in->data_sample(0,0);  // Get the first sample of the first channel
    
    // Copy timestamps to both output slots
    data_phase_out = data_out_port_->slot(0)->ClaimData(false);
    data_phase_out->CloneTimestamps(*data_in);
    data_real_out = data_out_port_->slot(1)->ClaimData(false);
    data_real_out->CloneTimestamps(*data_in);

    sample_window.push_back(sample);

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

      // Construct analytic signal spectrum
      construct_analytic_spectrum(n_fft, freq_half, freq);

      // IFFT
      fftwf_execute(p_inv);

      // Normalize the output of the inverse FFT
      for (int i = 0; i < n_fft; i++) {
        out[i][0] /= n_fft;
        out[i][1] /= n_fft;
      }

      // Get phase and real part of the last sample (N-1) of the original signal
      phase = std::atan2(out[N-1][1], out[N-1][0]);
      real_part = out[N-1][0];

      data_phase_out->set_data_sample(0,0, phase);
      data_real_out->set_data_sample(0,0, real_part);
    }

    data_out_port_->slot(0)->PublishData();    
    data_out_port_->slot(1)->PublishData();

    packet_count_++;

  }

  fftwf_destroy_plan(p);
  fftwf_destroy_plan(p_inv);
  fftwf_free(signal_in);
  fftwf_free(freq_half);
  fftwf_free(freq);
  fftwf_free(out);
}

void PhaseEstimator::Postprocess(ProcessingContext &context) {
  printf("\n PhaseEstimator:Total messages processed: %d", packet_count_);
}

REGISTERPROCESSOR(PhaseEstimator);
