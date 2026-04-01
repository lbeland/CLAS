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
#include <dsp/fftw_planner_mutex.hpp>

float round(float var, float precision)
{
    // 37.66666 * 100 =3766.66
    // 3766.66 + .5 =3767.16    for rounding off value
    // then type cast to int so value is 3767
    // then divided by 100 so the value converted into 37.67
    float value = (int)(var * (1.0f / precision) + 0.5f);
    return (float)value * precision;
}

PhaseEstimator::PhaseEstimator() : IProcessor(PRIORITY_HIGH) {
  add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
  add_option("n_fft", n_fft_, "FFT size");
  add_option("calibrate", calibrate_, "Whether to apply calibration gain");
  add_option("iaf_default", iaf_default_, "Default individual alpha frequency before shared state updates.");
  add_option("iaf_read_interval", iaf_read_interval_, "Packets between shared IAF polling steps.");
  add_option("filter", filter_def_, "Filter definition.", true);

  iaf_state_ = create_follower_state<float>(
      "iaf", iaf_default_(), Permission::NONE,
      "Individual alpha frequency shared by an upstream processor.");
}

void PhaseEstimator::Configure(const GlobalContext &context) {
  return;
}

void PhaseEstimator::CreatePorts() {
  data_in_port_ = create_input_port<MultiChannelType<float>>(
      "in", 
      MultiChannelType<float>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
      PortInPolicy(SlotRange(0,MAX_NCHANNELS)));

  data_out_port_ = create_output_port<MultiChannelType<float>>(
      "out",
      MultiChannelType<float>::Parameters(1,1,1), // Placeholder, will be set in CompleteStreamInfo
      PortOutPolicy(SlotRange(0,MAX_NCHANNELS),200,WaitStrategy::kBlockingStrategy));
}

void PhaseEstimator::CompleteStreamInfo() {
  const auto &input_info = data_in_port_->slot(0)->streaminfo();
  const auto &input_params =
      input_info.parameters<MultiChannelType<float>::Parameters>();

  for (int k = 0; k < data_out_port_->number_of_slots(); ++k) {
    data_out_port_->streaminfo(k).set_stream_rate(data_in_port_->streaminfo(0).stream_rate());
    dynamic_cast<StreamInfo<MultiChannelType<float>>&>(
      data_out_port_->slot(k)->streaminfo()).set_parameters(input_params);
  }

}

void PhaseEstimator::calibrate_gain(const int N) {
  // This function calculates the MSE-optimal calibration gain for cecHT based on the provided bandpass filter coefficients.

  if (calibrate_()) {
    printf("Calculating calibration gain for cecHT with N=%d, f0=%f, fs=%f\n", N, f0_, fs_);
    // Calculate calibration gain
    const int L = n_fft_();
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
          static_cast<double>(coeffs_[k].real()),
          static_cast<double>(coeffs_[k].imag())
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

void PhaseEstimator::load_filter_coeffs(const StorageContext& context) {
  const float iaf = iaf_state_->get();
  if (!filter_def_()["file"]) {
    int N = filter_def_()["N"].as<int>(1);
    float bandwith = filter_def_()["bandwidth"].as<float>(4.0);
    float low_cutoff = iaf - bandwith/2.0;
    float high_cutoff = iaf + bandwith/2.0;
    int window_size = filter_def_()["length"].as<int>(n_fft_());
    std::string filename;
    filename = std::to_string(N) + "_" + std::format("{:.1f}", low_cutoff) + "_" + std::format("{:.1f}", high_cutoff) + "_" + std::to_string(fs_) + "_" + std::to_string(window_size) + ".txt";      

    coeff_file_ = context.resolve_path(filename, "filters");
    
  } else {
    coeff_file_ = context.resolve_path(
        filter_def_()["file"].as<std::string>(), "filters");
  }

  coeffs_.clear();
  coeffs_.reserve(static_cast<size_t>(n_fft_()));

  // Load bandpass filter coefficients for cecHT from file
  std::ifstream stream(coeff_file_);

  if (!stream.good()) {
    throw std::runtime_error("PhaseEstimator: Cannot open filter coefficients file");
  }

  auto header = dsp::filter::parse_file_header(stream);

  if (header["type"] != "frequency response") {
    throw std::runtime_error("PhaseEstimator: Expected frequency response in file");
  }

  float real = 0.0f;
  float imag = 0.0f;
  while (stream >> real >> imag) {
    coeffs_.emplace_back(real, imag);
  }

  if (coeffs_.size() < static_cast<size_t>(n_fft_())) {
    throw std::runtime_error(
        "PhaseEstimator: coefficient file has fewer bins than n_fft");
  }
}

void PhaseEstimator::Prepare(GlobalContext &context) {
  const auto& info = data_in_port_->streaminfo(0);
  const auto& p = info.parameters<MultiChannelType<float>::Parameters>();
  LOG(INFO) << "Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate << "\n";
  fs_ = p.sample_rate;
  f0_ = iaf_state_->get();
  int N = static_cast<int>(fs_ * (1.0/f0_)*2.0);  // 2 cycles of a 10 Hz sine wave
  sample_window.set_capacity(N);
  LOG(INFO) << "Sample window size set to " << sample_window.capacity() << "\n";

  const int n_fft = static_cast<int>(n_fft_());
  if (n_fft < N) {
    throw std::runtime_error("PhaseEstimator: n_fft must be >= window length");
  }

  load_filter_coeffs(context);
  calibrate_gain(N);
}

void PhaseEstimator::construct_analytic_spectrum(int n_fft, const fftwf_complex* half, fftwf_complex* full){

  full[0][0] = half[0][0];
  full[0][1] = half[0][1];

  if (n_fft % 2 == 0) {
      for (int k = 1; k < n_fft / 2; ++k) {
          full[k][0] = 2.0 * half[k][0];
          full[k][1] = 2.0 * half[k][1];
      }

      full[n_fft / 2][0] = half[n_fft / 2][0];
      full[n_fft / 2][1] = half[n_fft / 2][1];

      for (int k = n_fft / 2 + 1; k < n_fft; ++k) {
          full[k][0] = 0.0;
          full[k][1] = 0.0;
      }
  } else {
      for (int k = 1; k <= (n_fft - 1) / 2; ++k) {
          full[k][0] = 2.0 * half[k][0];
          full[k][1] = 2.0 * half[k][1];
      }
      for (int k = (n_fft + 1) / 2; k < n_fft; ++k) {
          full[k][0] = 0.0;
          full[k][1] = 0.0;
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
  fftwf_complex* out = fftwf_alloc_complex(n_fft);

  fftwf_plan p;
  fftwf_plan p_inv;
  {
    std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
    p = fftwf_plan_dft_r2c_1d(n_fft, signal_in, freq_half, FFTW_ESTIMATE);
    p_inv = fftwf_plan_dft_1d(n_fft, freq, out, FFTW_BACKWARD, FFTW_ESTIMATE);
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

    if (iaf_read_interval_() > 0 && (packet_count_ % iaf_read_interval_()) == 0) {
      const float new_f0 = iaf_state_->get();
      printf("\n Packet %d: Read shared IAF value: %.2f Hz", packet_count_, new_f0);
      if (std::abs(new_f0 - f0_) > 0.05f) {  // Only update if IAF has changed by more than 0.05 Hz to avoid unnecessary recalibration
        f0_ = round(new_f0, 0.1f);  // Round to nearest 0.1 Hz for stability
        load_filter_coeffs(context);
        calibrate_gain(N);
      }
      else {
        printf(" - No need for calibration");
      }
    }
    // TimePoint start_time = Clock::now();

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
      // TimePoint start_time = Clock::now();

      // FFT
      fftwf_execute(p);

      // TimePoint end_time = Clock::now();
      // std::chrono::duration<double> elapsed = end_time - start_time;
      // printf("Processed packet %d in %.9f microseconds\n", packet_count_, elapsed.count()*1e6);

      // Construct analytic signal spectrum
      construct_analytic_spectrum(n_fft, freq_half, freq);

      // Multiply with Bandpass filter
      for (int k=0; k < n_fft; k++) {
        const float in_re = freq[k][0];
        const float in_im = freq[k][1];
        const float c_re = coeffs_[k].real();
        const float c_im = coeffs_[k].imag();

        freq[k][0] = in_re * c_re - in_im * c_im;
        freq[k][1] = in_re * c_im + in_im * c_re;
      }

      // IFFT
      fftwf_execute(p_inv);

      // Normalize the output of the inverse FFT and multiply with calibration gain
      for (int i = 0; i < n_fft; i++) {
        const float in_re = out[i][0];
        const float in_im = out[i][1];

        out[i][0] = (in_re * c_gain_.real() - in_im * c_gain_.imag()) / n_fft;
        out[i][1] = (in_re * c_gain_.imag() + in_im * c_gain_.real()) / n_fft;
      }

      // Get phase and real part of the last sample (N-1) of the original signal
      phase = std::atan2(out[N-1][1], out[N-1][0]);
      real_part = out[N-1][0];

      data_phase_out->set_data_sample(0,0, phase);
      data_real_out->set_data_sample(0,0, real_part);


    // TimePoint end_time = Clock::now();
    // std::chrono::duration<double> elapsed = end_time - start_time;
    // printf("Processed packet %d in %.9f microseconds\n", packet_count_, elapsed.count()*1e6);

    }

    data_out_port_->slot(0)->PublishData();    
    data_out_port_->slot(1)->PublishData();

    packet_count_++;
  }

  {
    std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
    fftwf_destroy_plan(p);
    fftwf_destroy_plan(p_inv);
  }
  fftwf_free(signal_in);
  fftwf_free(freq_half);
  fftwf_free(freq);
  fftwf_free(out);
}

void PhaseEstimator::Postprocess(ProcessingContext &context) {
  printf("\n ---------------- \n PhaseEstimator: Total messages processed: %d", packet_count_);
}

REGISTERPROCESSOR(PhaseEstimator);
