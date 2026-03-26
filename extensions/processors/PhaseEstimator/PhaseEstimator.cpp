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
#include <sstream>
#include <string>
#include <fftw3.h>
#include <complex>


PhaseEstimator::PhaseEstimator() : IProcessor(PRIORITY_HIGH) {
  add_option("n_messages", n_messages_, "Number of packets to receive");
  add_option("n_fft", n_fft_, "FFT size");
  add_option("calibrate", calibrate_, "Whether to apply calibration gain");
  add_option("coeff_file", coeff_file_, "Path to bandpass filter coefficients file");
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
  const auto& info = data_in_port_->streaminfo(0);
  const auto& p = info.parameters<MultiChannelType<float>::Parameters>();
  LOG(INFO) << "Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate << "\n";
  fs_ = p.sample_rate;
  f0_ = 10.0;
  int N = static_cast<int>(fs_ * (1.0/f0_)*2.0);  // 2 cycles of a 10 Hz sine wave
  sample_window.set_capacity(N);
  LOG(INFO) << "Sample window size set to " << sample_window.capacity() << "\n";

  const int n_fft = static_cast<int>(n_fft_());
  if (n_fft < N) {
    throw std::runtime_error("PhaseEstimator: n_fft must be >= window length");
  }

  coeffs_.reserve(static_cast<size_t>(n_fft));

  // Load bandpass filter coefficients for cecHT from file
  std::string coeff_file = context.resolve_path(coeff_file_());
  std::ifstream coeffs;
  if (!coeffs.is_open()) {
    coeffs.open(coeff_file);
  }
  if (!coeffs.is_open()) {
    throw std::runtime_error("PhaseEstimator: could not open coefficients file");
  }

  std::string line;
  size_t line_number = 0;
  while (std::getline(coeffs, line)) {
    ++line_number;
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    float real = 0.0f;
    float imag = 0.0f;
    char comma = '\0';
    if (!(ss >> real >> comma >> imag) || comma != ',') {
      throw std::runtime_error(
          "PhaseEstimator: invalid coefficient format at line " + std::to_string(line_number));
    }
    

    coeffs_.emplace_back(real, imag);
  }

  if (calibrate_()) {
    LOG(INFO) << "Calculating calibration gain...";

    // Calculate calibration gain
    const int L = n_fft;
    const int n = N - 1;
    const double PI = std::acos(-1.0);
    const double omega0 = 2.0 * PI * f0_ / fs_;

    // Lambda for Dirichlet kernel: D_N(alpha) = sin(N*alpha/2) / sin(alpha/2) * exp(i*alpha*(N-1)/2)
    auto dirichlet_N = [N](double alpha) -> std::complex<double> {
      if (std::abs(alpha) < 1e-12) {
        return std::complex<double>(static_cast<double>(N), 0.0);
      }
      double numerator = std::sin(0.5 * N * alpha);
      double denominator = std::sin(0.5 * alpha);
      if (std::abs(denominator) < 1e-12) {
        return std::complex<double>(static_cast<double>(N), 0.0);
      }
      double phase_angle = alpha * (N - 1) * 0.5;
      double mag = numerator / denominator;
      return std::complex<double>(mag * std::cos(phase_angle), mag * std::sin(phase_angle));
    };

    std::complex<double> P_sum(0.0, 0.0);
    std::complex<double> M_sum(0.0, 0.0);

    std::vector<std::complex<float>> shifted_coeffs_(n_fft);
    ifftshift(coeffs_, shifted_coeffs_, L);

    for (int k = 0; k < L; ++k) {
      double omega_k = 2.0 * PI * k / L;

      // Compute Dirichlet kernels for +/- frequency components
      std::complex<double> D_plus = dirichlet_N(omega0 - omega_k);
      std::complex<double> D_minus = dirichlet_N(-omega0 - omega_k);

      // X components (half-weighted)
      std::complex<double> X_plus = 0.5 * D_plus;
      std::complex<double> X_minus = 0.5 * D_minus;

      // Hilbert multiplier: h[0]=1, h[Nyquist]=1 (if even L), h[other positive]=2
      double h_mult = 1.0;
      if (k > 0 && k < (L / 2)) {
        h_mult = 2.0;
      } else if (k == L / 2 && L % 2 == 0) {
        h_mult = 1.0;
      }

      // Phase rotation for the n-th sample
      std::complex<double> phase_exp(0.0, omega_k * n);
      phase_exp = std::exp(phase_exp);

      // Combine Hilbert multiplier with loaded bandpass coefficients
      std::complex<double> coeff_k = std::complex<double>(
          static_cast<double>(shifted_coeffs_[k].real()),
          static_cast<double>(shifted_coeffs_[k].imag())
      );
      std::complex<double> G = h_mult * coeff_k;

      // Accumulate weighted sums
      P_sum += G * X_plus * phase_exp;
      M_sum += G * X_minus * phase_exp;
    }

    // Normalize by FFT size
    P_sum /= static_cast<double>(L);
    M_sum /= static_cast<double>(L);

    // Apply frequency shift to get analytic-aligned components
    std::complex<double> phase_shift(0.0, -omega0 * n);
    phase_shift = std::exp(phase_shift);
    std::complex<double> Gplus = P_sum * phase_shift;
    std::complex<double> Gminus = M_sum * phase_shift;

    // Calculate MSE-optimal calibration gain: C_opt = conj(Gplus) / (|Gplus|^2 + |Gminus|^2)
    double denom = std::norm(Gplus) + std::norm(Gminus);
    if (denom > 1e-12) {
      std::complex<double> C_opt = std::conj(Gplus) / denom;
      c_gain_ = std::complex<float>(static_cast<float>(C_opt.real()),
                                    static_cast<float>(C_opt.imag()));
    } else {
      c_gain_ = std::complex<float>(1.0f, 0.0f);  // fallback to unity gain
    }
  } else {
    c_gain_ = std::complex<float>(1.0f, 0.0f);  // no calibration, unity gain
  }
  LOG(INFO) << "Calibration gain set to: " << c_gain_.real() << " + " << c_gain_.imag() << "i\n";
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

void PhaseEstimator::ifftshift(const std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out, int L)
{
    int s = (L + 1) / 2;  // ceil(L/2)
    for (int k = 0; k < L; ++k) {
        int src = (k + s) % L;
        out[k] = in[src];
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
  fftwf_complex* freq_shifted = fftwf_alloc_complex(n_fft);
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

      // Shift the spectrum so that the DC component is at the center
      fftshift(freq, freq_shifted, n_fft);

      // Multiply with Bandpass filter
      for (int k=0; k < n_fft; k++) {
        const float in_re = freq_shifted[k][0];
        const float in_im = freq_shifted[k][1];
        const float c_re = coeffs_[k].real();
        const float c_im = coeffs_[k].imag();

        freq_shifted[k][0] = in_re * c_re - in_im * c_im;
        freq_shifted[k][1] = in_re * c_im + in_im * c_re;
      }
      ifftshift(freq_shifted, freq, n_fft);

      // IFFT
      fftwf_execute(p_inv);

      // Normalize the output of the inverse FFT and multiply with calibration gain
      for (int i = 0; i < n_fft; i++) {
        const float in_re = out[i][0];
        const float in_im = out[i][1];

        out[i][0] = (in_re * c_gain_.real() - in_im * c_gain_.imag()) / n_fft; //in_re / n_fft; //
        out[i][1] = (in_re * c_gain_.imag() + in_im * c_gain_.real()) / n_fft; //in_im / n_fft; //
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
  fftwf_free(freq_shifted);
  fftwf_free(freq);
  fftwf_free(out);
}

void PhaseEstimator::Postprocess(ProcessingContext &context) {
  printf("\n PhaseEstimator:Total messages processed: %d", packet_count_);
}

REGISTERPROCESSOR(PhaseEstimator);
