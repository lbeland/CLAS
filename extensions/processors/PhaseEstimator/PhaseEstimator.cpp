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

namespace
{
    size_t good_size_real(size_t n)
    // from https://github.com/hayguen/pocketfft/blob/cpp/pocketfft_hdronly.h
    {
        if (n <= 6)
            return n;

        size_t bestfac = 2 * n;
        for (size_t f5 = 1; f5 < bestfac; f5 *= 5)
        {
            size_t x = f5;
            while (x < n)
                x *= 2;
            for (;;)
            {
                if (x < n)
                    x *= 3;
                else if (x > n)
                {
                    if (x < bestfac)
                        bestfac = x;
                    if (x & 1)
                        break;
                    x >>= 1;
                }
                else
                    return n;
            }
        }
        return bestfac;
    }

    double round(double var, double precision)
    {
        // 37.66666 * 100 =3766.66
        // 3766.66 + .5 =3767.16    for rounding off value
        // then type cast to int so value is 3767
        // then divided by 100 so the value converted into 37.67
        double value = (int)(var * (1.0 / precision) + 0.5);
        return (double)value * precision;
    }

    void fftshift(const fftwf_complex *in, fftwf_complex *out, int L)
    {
        int s = L / 2; // floor(L/2)
        for (int k = 0; k < L; ++k)
        {
            int src = (k + s) % L;
            out[k][0] = in[src][0];
            out[k][1] = in[src][1];
        }
    }

    void ifftshift(const fftwf_complex *in, fftwf_complex *out, int L)
    {
        int s = (L + 1) / 2; // ceil(L/2)
        for (int k = 0; k < L; ++k)
        {
            int src = (k + s) % L;
            out[k][0] = in[src][0];
            out[k][1] = in[src][1];
        }
    }

    void ifftshift(const std::vector<std::complex<float>> &in, std::vector<std::complex<float>> &out, int L)
    {
        int s = (L + 1) / 2; // ceil(L/2)
        for (int k = 0; k < L; ++k)
        {
            int src = (k + s) % L;
            out[k] = in[src];
        }
    }

    void construct_analytic_spectrum(int n_fft, const fftwf_complex *half, fftwf_complex *full)
    {
        full[0][0] = half[0][0];
        full[0][1] = half[0][1];

        if (n_fft % 2 == 0)
        {
            for (int k = 1; k < n_fft / 2; ++k)
            {
                full[k][0] = 2.0 * half[k][0];
                full[k][1] = 2.0 * half[k][1];
            }

            full[n_fft / 2][0] = half[n_fft / 2][0];
            full[n_fft / 2][1] = half[n_fft / 2][1];

            for (int k = n_fft / 2 + 1; k < n_fft; ++k)
            {
                full[k][0] = 0.0;
                full[k][1] = 0.0;
            }
        }
        else
        {
            for (int k = 1; k <= (n_fft - 1) / 2; ++k)
            {
                full[k][0] = 2.0 * half[k][0];
                full[k][1] = 2.0 * half[k][1];
            }
            for (int k = (n_fft + 1) / 2; k < n_fft; ++k)
            {
                full[k][0] = 0.0;
                full[k][1] = 0.0;
            }
        }
    }

}

PhaseEstimator::PhaseEstimator() : IProcessor(PRIORITY_HIGH)
{
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    // add_option("n_fft", n_fft_, "FFT size");
    add_option("calibrate", calibrate_, "Whether to apply calibration gain");
    add_option("iaf_read_interval", iaf_read_interval_, "Packets between shared IAF polling steps.");
    add_option("filter", filter_def_, "Filter definition.", true);
    add_option("compensate_filter", compensate_filter_, "Whether to compensate the phase distortion of the preceding bandpass filter.", true);

    iaf_state_ = create_follower_state<double>(
        "iaf", std::numeric_limits<double>::quiet_NaN(), Permission::NONE,
        "Individual alpha frequency shared by an upstream processor.");
}

void PhaseEstimator::CreatePorts()
{
    data_in_port_ = create_input_port<MultiChannelType<float>>(
        "in",
        MultiChannelType<float>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)), // Accept only one channel
        PortInPolicy(SlotRange(0, MAX_NCHANNELS)));

    data_out_port_ = create_output_port<MultiChannelType<double>>(
        "out",
        MultiChannelType<double>::Parameters(1, 1, 1), // Placeholder, will be set in CompleteStreamInfo
        PortOutPolicy(SlotRange(0, MAX_NCHANNELS), 200, WaitStrategy::kBlockingStrategy));
}

void PhaseEstimator::CompleteStreamInfo()
{
    const auto &input_params = data_in_port_->slot(0)->streaminfo().parameters<MultiChannelType<float>::Parameters>();

    for (int k = 0; k < data_out_port_->number_of_slots(); ++k)
    {
        data_out_port_->streaminfo(k).set_parameters(input_params);
        data_out_port_->streaminfo(k).set_stream_rate(data_in_port_->streaminfo(0).stream_rate());
    }
}

void PhaseEstimator::calibrate_gain(const int N)
{
    // This function calculates the MSE-optimal calibration gain for cecHT based on the provided bandpass filter coefficients.

    if (calibrate_())
    {
        // Calculate calibration gain
        const int L = n_fft_;
        const int n = N - 1;
        const double omega0 = 2.0 * M_PI * f0_ / fs_;

        // Lambda for Dirichlet kernel: D_N(alpha) = sin(N*alpha/2) / sin(alpha/2) * exp(i*alpha*(N-1)/2)
        auto dirichlet_N = [N](double alpha) -> std::complex<double>
        {
            if (std::abs(alpha) < 1e-12)
            {
                return std::complex<double>(static_cast<double>(N), 0.0);
            }
            double numerator = std::sin(0.5 * N * alpha);
            double denominator = std::sin(0.5 * alpha);
            if (std::abs(denominator) < 1e-12)
            {
                return std::complex<double>(static_cast<double>(N), 0.0);
            }
            double phase_angle = alpha * (N - 1) * 0.5;
            double mag = numerator / denominator;
            return std::complex<double>(mag * std::cos(phase_angle), mag * std::sin(phase_angle));
        };

        std::complex<double> P_sum(0.0, 0.0);
        std::complex<double> M_sum(0.0, 0.0);

        for (int k = 0; k < L; ++k)
        {
            double omega_k = 2.0 * M_PI * k / L;

            // Compute Dirichlet kernels for +/- frequency components
            std::complex<double> D_plus = dirichlet_N(omega0 - omega_k);
            std::complex<double> D_minus = dirichlet_N(-omega0 - omega_k);

            // X components (half-weighted)
            std::complex<double> X_plus = 0.5 * D_plus;
            std::complex<double> X_minus = 0.5 * D_minus;

            // Hilbert multiplier: h[0]=1, h[Nyquist]=1 (if even L), h[other positive]=2
            double h_mult = 1.0;
            if (k > 0 && k < (L / 2))
            {
                h_mult = 2.0;
            }
            else if (k == L / 2 && L % 2 == 0)
            {
                h_mult = 1.0;
            }

            // Phase rotation for the n-th sample
            std::complex<double> phase_exp(0.0, omega_k * n);
            phase_exp = std::exp(phase_exp);

            // Combine Hilbert multiplier with loaded bandpass coefficients
            std::complex<double> coeff_k = std::complex<double>(
                static_cast<double>(coeffs_[k].real()),
                static_cast<double>(coeffs_[k].imag()));
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
        if (denom > 1e-12)
        {
            std::complex<double> C_opt = std::conj(Gplus) / denom;
            c_gain_ = std::complex<double>(static_cast<double>(C_opt.real()),
                                          static_cast<double>(C_opt.imag()));
        }
        else
        {
            c_gain_ = std::complex<double>(1.0, 0.0); // fallback to unity gain
        }
        
        LOG(INFO) << "Calibration gain for " << f0_ << " Hz set to: " << c_gain_.real() << " + " << c_gain_.imag() << "i";
    }
    else
    {
        c_gain_ = std::complex<double>(1.0, 0.0); // no calibration, unity gain
    }
}

void PhaseEstimator::load_filter_coeffs(const StorageContext &context, double iaf)
{
    if (!filter_def_()["file"])
    {
        int N = filter_def_()["N"].as<int>(1);
        double bandwidth = filter_def_()["bandwidth"].as<double>(4.0);
        double low_cutoff = iaf - bandwidth / 2.0;
        double high_cutoff = iaf + bandwidth / 2.0;
        int window_size = n_fft_; // next fast len for 2 cycles of iaf frequency
        std::string filename;
        filename = std::to_string(N) + "_" + std::format("{:.2f}", low_cutoff) + "_" + std::format("{:.2f}", high_cutoff) + "_" + std::to_string(fs_) + "_" + std::to_string(window_size) + ".txt";

        coeff_file_ = context.resolve_path(filename, "filters");
    }
    else
    {
        coeff_file_ = context.resolve_path(
            filter_def_()["file"].as<std::string>(), "filters");
    }

    coeffs_.clear();
    coeffs_.reserve(static_cast<size_t>(n_fft_));

    // Load bandpass filter coefficients for cecHT from file
    std::ifstream stream(coeff_file_);

    if (!stream.good())
    {
        throw std::runtime_error("PhaseEstimator: Cannot open filter coefficients file: " + coeff_file_);
    }

    auto header = dsp::filter::parse_file_header(stream);

    if (header["type"] != "frequency response")
    {
        throw std::runtime_error("PhaseEstimator: Expected frequency response in file");
    }

    double real = 0.0;
    double imag = 0.0;
    while (stream >> real >> imag)
    {
        coeffs_.emplace_back(real, imag);
    }

    if (coeffs_.size() != static_cast<size_t>(n_fft_))
    {
        throw std::runtime_error(
            "PhaseEstimator: bandpass coefficient file has " + std::to_string(coeffs_.size()) + " bins, expected " + std::to_string(n_fft_));
    }
}


void PhaseEstimator::load_phase_shift(const StorageContext &context, double iaf)
{
    std::string filename;

    filename =  "global_filter_" + std::to_string(fs_) + "_phase.txt";

    std::string phase_shift_file_ = context.resolve_path(filename, "filters");

    filter_phase_shift_values.clear();

    // Load bandpass filter coefficients for cecHT from file
    std::ifstream stream(phase_shift_file_);

    if (!stream.good())
    {
        throw std::runtime_error("PhaseEstimator: Cannot open phase shift file: " + phase_shift_file_);
    }

    auto header = dsp::filter::parse_file_header(stream);

    if (header["type"] != "phase shift")
    {
        throw std::runtime_error("PhaseEstimator: Expected phase shift in file:" + phase_shift_file_);
    }

    double phase_shift_rad;
    while (stream >> phase_shift_rad)
    {
        filter_phase_shift_values.emplace_back(phase_shift_rad);
    }
    filter_phase_shift_ = filter_phase_shift_values[iaf * 10]; // Phase shift values are stored in 0.1 Hz increments and thus indexed with (iaf * 10)
    LOG(INFO) << name() << " Loaded phase shift of " << filter_phase_shift_ << " radians for compensation";
    
}

void PhaseEstimator::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);
    const auto &p = info.parameters<MultiChannelType<float>::Parameters>();
    LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate;
    fs_ = p.sample_rate;
}

void PhaseEstimator::Preprocess(ProcessingContext &context)
{
    packet_count_ = 0;

    window_size_ = static_cast<int>(fs_ * (1.0 / f0_) * 2.0); // 2 cycles of a 10 Hz sine wave
    n_fft_ = static_cast<int>(good_size_real(window_size_));
    sample_window.set_capacity(static_cast<int>(fs_ * (1.0 / 5.0) * 2.0)); // Initialize with max expected window size for 5 Hz IAF
    LOG(INFO) << name() << " Sample window size set to " << window_size_ << ", FFT size: " << n_fft_;

    load_filter_coeffs(context, f0_);
    calibrate_gain(window_size_);

    if (compensate_filter_())
    {
        LOG(INFO) << name() << " Loading phase shift values for filter compensation";
        load_phase_shift(context, f0_);
    }

    signal_in = fftwf_alloc_real(n_fft_);
    freq_half = fftwf_alloc_complex(n_fft_ / 2 + 1);
    freq = fftwf_alloc_complex(n_fft_);
    out = fftwf_alloc_complex(n_fft_);

    // Import FFTW wisdom for optimal FFT planning
    fftwf_import_wisdom_from_filename((context.resolve_path("fftw_wisdom.txt", "fft_wisdom")).c_str());

    {
        std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
        p_ = fftwf_plan_dft_r2c_1d(n_fft_, signal_in, freq_half,  FFTW_WISDOM_ONLY);
        if (p_ == nullptr)
        {
            LOG(WARNING) << name() << " No wisdom available for FFT planning, using patient mode.";
            p_ = fftwf_plan_dft_r2c_1d(n_fft_, signal_in, freq_half,  FFTW_PATIENT);
        }
        p_inv_ = fftwf_plan_dft_1d(n_fft_, freq, out, FFTW_BACKWARD,  FFTW_WISDOM_ONLY);
        if (p_inv_ == nullptr)
        {
            LOG(WARNING) << name() << " No wisdom available for IFFT planning, using patient mode.";
            p_inv_ = fftwf_plan_dft_1d(n_fft_, freq, out, FFTW_BACKWARD,  FFTW_PATIENT);
        }
    }
}

void PhaseEstimator::Process(ProcessingContext &context)
{
    MultiChannelType<float>::Data *data_in;
    MultiChannelType<double>::Data *data_real_out = nullptr;
    MultiChannelType<double>::Data *data_phase_out = nullptr;

    // FFTW output spectrum
    float sample;
    double phase;
    double real_part;

    // Measurement phase
    while (!context.terminated())
    {

        if (n_messages_() != -1 && packet_count_ >= static_cast<size_t>(n_messages_()))
        {
            break;
        }

        // Try to retrieve data
        if (!data_in_port_->slot(0)->RetrieveData(data_in))
        {
            break;
        }

        sample = data_in->data_sample(0, 0); // Get the first sample of the first channel

        // Copy timestamps to both output slots
        data_phase_out = data_out_port_->slot(0)->ClaimData(false);
        data_phase_out->CloneTimestamps(*data_in);
        data_real_out = data_out_port_->slot(1)->ClaimData(false);
        data_real_out->CloneTimestamps(*data_in);

        sample_window.push_back(sample);

        data_in_port_->slot(0)->ReleaseData();

        TimePoint start_time = Clock::now();

        if (packet_count_ % iaf_read_interval_() == 0)
        {
            double new_f0 = iaf_state_->get();
            if (std::isnan(new_f0))
            {
                valid_iaf_ = false;
            }
            else
            {
                valid_iaf_ = true;
                new_f0 = round(new_f0, 0.1); // Round to nearest 0.1 Hz to avoid excessive recalibration due to small IAF fluctuations
            
                // printf("\n Packet %d: Read shared IAF value: %.2f Hz", packet_count_, new_f0);
                if (std::abs(new_f0 - f0_) >= 0.1)
                {                              
                    // Only update if IAF has changed by more than 0.1 Hz to avoid unnecessary recalibration
                    f0_ = new_f0;
                    int old_n_fft = n_fft_; // Store old FFT size to check if we need to reallocate FFTW arrays
                    window_size_ = static_cast<int>(2.0 * fs_ / f0_);
                    if (window_size_ > sample_window.capacity())
                    {
                        LOG(WARNING) << name() << " New window size " << window_size_ << " exceeds circular buffer capacity " << sample_window.capacity() << ". Resizing circular buffer to new window size.";
                        sample_window.rset_capacity(window_size_);
                    }
                    n_fft_ = good_size_real(window_size_);
                    LOG(INFO) << name() << " Packet " << packet_count_ << ": Update IAF to " << f0_ << " Hz, window size: " << window_size_ << ", FFT size: " << n_fft_;

                    load_filter_coeffs(context, f0_);
                    calibrate_gain(window_size_);
                    if (old_n_fft != n_fft_)
                    {
                        // Reallocate FFTW arrays with new size
                        {
                            std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
                            fftwf_destroy_plan(p_);
                            fftwf_destroy_plan(p_inv_);
                        }
                        fftwf_free(signal_in);
                        fftwf_free(freq_half);
                        fftwf_free(freq);
                        fftwf_free(out);
                        
                        signal_in = fftwf_alloc_real(n_fft_);
                        freq_half = fftwf_alloc_complex(n_fft_ / 2 + 1);
                        freq = fftwf_alloc_complex(n_fft_);
                        out = fftwf_alloc_complex(n_fft_);
                        {
                            std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
                            p_ = fftwf_plan_dft_r2c_1d(n_fft_, signal_in, freq_half,  FFTW_WISDOM_ONLY);
                            if (p_ == nullptr)
                            {
                                LOG(WARNING) << name() << "No wisdom available for FFT planning, using estimate mode.";
                                p_ = fftwf_plan_dft_r2c_1d(n_fft_, signal_in, freq_half,  FFTW_ESTIMATE);
                            }
                            p_inv_ = fftwf_plan_dft_1d(n_fft_, freq, out, FFTW_BACKWARD,  FFTW_WISDOM_ONLY);
                            if (p_inv_ == nullptr)
                            {
                                LOG(WARNING) << name() << "No wisdom available for IFFT planning, using estimate mode.";
                                p_inv_ = fftwf_plan_dft_1d(n_fft_, freq, out, FFTW_BACKWARD,  FFTW_ESTIMATE);
                            }
                        }
                    }

                    // Update value for phase shift compensation
                    if (compensate_filter_())
                    {
                        filter_phase_shift_ = filter_phase_shift_values[f0_ * 10]; // Phase shift values are stored in 0.1 Hz increments and thus indexed with (iaf * 10)
                        LOG(INFO) << name() << " Loaded phase shift of " << filter_phase_shift_ << " radians for compensation";
                    }
                }                
                // else {
                //   printf(" - No need for calibration");
                // }
            }
        }

        TimePoint claim_output_time = Clock::now();

        if (valid_iaf_ && sample_window.size() >= static_cast<size_t>(window_size_))
        { //} && (packet_count_ % (sample_window.capacity()/2) == 0)) {

            // Convert circular buffer<float> to continuous array for FFTW input and zero-pad to n_fft length
            // signal_in = sample_window.linearize();
            auto start = sample_window.end() - window_size_;
            for (int i = 0; i < window_size_; ++i)
            {
                signal_in[i] = start[i]; // Get the last 'window_size_' samples from the circular buffer
            }
            for (int i = window_size_; i < n_fft_; ++i)
            {
                signal_in[i] = 0.0f;
            }
            TimePoint copy_buffer_time = Clock::now();

            // FFT
            fftwf_execute(p_);

            TimePoint fft_time = Clock::now();
            // std::chrono::duration<double> elapsed = end_time - start_time;
            // printf("Processed packet %d in %.9f microseconds\n", packet_count_, elapsed.count()*1e6);

            // Construct analytic signal spectrum
            construct_analytic_spectrum(n_fft_, freq_half, freq);

            TimePoint ana_spec_time = Clock::now();

            // Multiply with Bandpass filter
            for (int k = 0; k < n_fft_; k++)
            {
                const double in_re = freq[k][0];
                const double in_im = freq[k][1];
                const double c_re = coeffs_[k].real();
                const double c_im = coeffs_[k].imag();
                freq[k][0] = in_re * c_re - in_im * c_im;
                freq[k][1] = in_re * c_im + in_im * c_re;
            }

            TimePoint filter_time = Clock::now();

            // IFFT
            fftwf_execute(p_inv_);

            TimePoint ifft_time = Clock::now();

            // Normalize the output of the inverse FFT and multiply with calibration gain
            double c_gain_re = c_gain_.real();
            double c_gain_im = c_gain_.imag();
            for (int i = 0; i < n_fft_; i++)
            {
                const double in_re = out[i][0];
                const double in_im = out[i][1];

                out[i][0] = (in_re * c_gain_re - in_im * c_gain_im) / n_fft_;
                out[i][1] = (in_re * c_gain_im + in_im * c_gain_re) / n_fft_;
            }

            // Get phase and real part of the last sample
            phase = std::atan2(out[window_size_-1][1], out[window_size_-1][0]);

            // Compensate for phase distortion of the preceding bandpass filter if enabled
            phase -= filter_phase_shift_;

            // LOG(INFO) << "estimated phase: " << phase << " radians, " << (phase * 180.0f / M_PI) << " degrees";
            real_part = out[window_size_-1][0];

            TimePoint norm_cal_time = Clock::now();

            // data_phase_out->set_source_timestamp(Clock::now());

            data_phase_out->set_data_sample(0, 0, phase);
            data_phase_out->set_sample_timestamps(data_in->sample_timestamps());
            data_real_out->set_data_sample(0, 0, real_part);
            data_real_out->set_sample_timestamps(data_in->sample_timestamps());

            TimePoint end_time = Clock::now();

            // LOG(INFO) << name() << " Processed packet " << packet_count_ << " - timings (us): 
            //           << " claim_output=" << std::chrono::duration<double, std::micro>(claim_output_time - start_time).count()
            //           << ", copy_buffer=" << std::chrono::duration<double, std::micro>(copy_buffer_time - claim_output_time).count()
            //           << ", fft=" << std::chrono::duration<double, std::micro>(fft_time - copy_buffer_time).count()
            //           << ", ana_spec=" << std::chrono::duration<double, std::micro>(ana_spec_time - fft_time).count()
            //           << ", filter=" << std::chrono::duration<double, std::micro>(filter_time - ana_spec_time).count()
            //           << ", ifft=" << std::chrono::duration<double, std::micro>(ifft_time - filter_time).count()
            //           << ", norm_cal=" << std::chrono::duration<double, std::micro>(norm_cal_time - ifft_time).count()
            //           << ", total=" << std::chrono::duration<double, std::micro>(end_time - start_time).count();

            // double processing_time_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
            // LOG(INFO) << name() << " Processed packet "<< packet_count_ << " in " << std::fixed << std::setprecision(4) << processing_time_us << " us";
        }
        else
        {
            // Set to NaN to indicate invalid IAF
            data_phase_out->set_data_sample(0, 0, std::numeric_limits<double>::quiet_NaN());
            data_real_out->set_data_sample(0, 0, std::numeric_limits<double>::quiet_NaN()); 
        }

        data_out_port_->slot(0)->PublishData();
        data_out_port_->slot(1)->PublishData();

        packet_count_++;
    }

}

void PhaseEstimator::Postprocess(ProcessingContext &context)
{
    printf("\n ---------------- \n PhaseEstimator: Total messages processed: %d", packet_count_);

}

void PhaseEstimator::Unprepare(GlobalContext &context)
{ 
    // Save FFTW wisdom for future runs to speed up plan creation
    int ret = fftwf_export_wisdom_to_filename(context.resolve_path("fftw_wisdom.txt", "fft_wisdom").c_str());
    if (ret == 0)
    {
        LOG(WARNING) << name() << " Failed to save FFTW wisdom to file.";
    }
    {
        std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
        fftwf_destroy_plan(p_);
        fftwf_destroy_plan(p_inv_);
    }
    fftwf_free(signal_in);
    fftwf_free(freq_half);
    fftwf_free(freq);
    fftwf_free(out);
}

REGISTERPROCESSOR(PhaseEstimator);