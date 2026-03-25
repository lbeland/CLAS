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
  sample_window.set_capacity(fs_ * (1.0/10.0)*2.0);  // 2 cycles of a 10 Hz sine wave
  printf("Sample window size set to %lu\n", sample_window.capacity());
}

int PhaseEstimator::get_max_bin(fftw_complex* out, size_t out_size) {
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

void PhaseEstimator::fftshift(const fftw_complex* in, fftw_complex* out, int L) {
    int s = L / 2;  // floor(L/2)
    for (int k = 0; k < L; ++k) {
        int src = (k + s) % L;
        out[k][0] = in[src][0];
        out[k][1] = in[src][1];
    }
}
void PhaseEstimator::ifftshift(const fftw_complex* in, fftw_complex* out, int L) {
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

  int f0_ = 10;

  // FFTW output spectrum
  float sample;
  float phase;
  float real_part;
  const int N = static_cast<int>(sample_window.capacity());
  fftw_complex* complex_in = fftw_alloc_complex(N);
  fftw_complex* freq = fftw_alloc_complex(N);
  fftw_complex* out = fftw_alloc_complex(N);

  fftw_plan p_inv = fftw_plan_dft_1d(N, freq, out, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_plan p = fftw_plan_dft_1d(N, complex_in, freq, FFTW_FORWARD, FFTW_ESTIMATE);

  // Set imaginary part of input to zero
  for (int i = 0; i < N; i++) {
    complex_in[i][1] = 0.0;
  }

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

    
    data_phase_out = data_out_port_->slot(0)->ClaimData(false);
    data_phase_out->CloneTimestamps(*data_in);
    data_real_out = data_out_port_->slot(1)->ClaimData(false);
    data_real_out->CloneTimestamps(*data_in);

    sample_window.push_back(sample);

    data_in_port_->slot(0)->ReleaseData();

    if ((sample_window.size() == sample_window.capacity())) { //} && (packet_count_ % (sample_window.capacity()/2) == 0)) {

      // Convert circular buffer<float> to contiguous std::vector<double>
      // std::vector<double> in(sample_window.begin(), sample_window.end());

      for (int n = 0; n < N; ++n) {
        complex_in[n][0] = static_cast<double>(sample_window[n]);
      }

      fftw_execute(p);

      int target_bin = static_cast<int>(std::round(f0_ * N / fs_));
      int max_bin = get_max_bin(freq, N);
      // if (max_bin != target_bin) {
      //   std::cout << "\nPacket_count: " << packet_count_ << ", Expected 10 Hz bin: " << target_bin << '\n';
      //   std::cout << "Maximum magnitude bin: " << max_bin
      //             << " (" << max_bin * fs_ / N << " Hz)\n";

      // Create analytic signal
      // Build analytic signal spectrum
      // DC - do nothing
      // freq[0][0] *= 1.0;
      // freq[0][1] *= 1.0;

      int pos_end;   // last positive-frequency bin to double
      int neg_start; // first negative-frequency bin to zero

      if (N % 2 == 0) {
          // even N
          freq[N/2][0] = static_cast<double>(sample_window[N/2]); // Nyquist - do nothing
          freq[N/2][1] = 1.0;
          pos_end = N/2 - 1;
          neg_start = N/2 + 1;
      } else {
          // odd N
          pos_end = (N - 1)/2;
          neg_start = (N + 1)/2;
      }

      for (int k = 1; k <= pos_end; ++k) {
          freq[k][0] = 2.0 * static_cast<double>(sample_window[k]);
          freq[k][1] = 2.0 * static_cast<double>(sample_window[k]);
      }

      for (int k = neg_start; k < N; ++k) {
          freq[k][0] = 0.0;
          freq[k][1] = 0.0;
      }

      fftw_execute(p_inv);

      // Normalize the output of the inverse FFT
      for (int i = 0; i < N; i++) {
        out[i][0] /= N;
        out[i][1] /= N;
      }

      phase = std::atan2(out[N-1][1], out[N-1][0]);
      real_part = out[N-1][0];

      data_phase_out->set_data_sample(0,0, phase);
      data_real_out->set_data_sample(0,0, real_part);
    }

    data_out_port_->slot(0)->PublishData();    
    data_out_port_->slot(1)->PublishData();

    packet_count_++;

  }

  fftw_free(complex_in);
  fftw_free(freq);
  fftw_free(out);
  fftw_destroy_plan(p);
  fftw_destroy_plan(p_inv);
}

void PhaseEstimator::Postprocess(ProcessingContext &context) {
  printf("\n PhaseEstimator:Total messages processed: %d", packet_count_);
}

REGISTERPROCESSOR(PhaseEstimator);
