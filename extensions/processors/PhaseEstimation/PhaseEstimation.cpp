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

#include "PhaseEstimation.hpp"
#include "utilities/time.hpp"
#include "logging/log.hpp"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <limits>
#include <sstream>
#include <string>
#include <algorithm>
#include <cstring>
#include <fftw3.h>
#include <complex>
#include <cmath>
#include <dsp/fftw_planner_mutex.hpp>

namespace
{
    // Returns the next FFT-efficient size >= n (favors factors of 2, 3, 5).
    // https://github.com/mreineck/ducc/blob/ducc0/src/ducc0/fft/fft.h
    // same as used by scipy.fftpack.next_fast_len
    size_t good_size_real(size_t n)
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

    // Round var to the nearest multiple of precision
    double round(double var, double precision)
    {
        double value = (int)(var * (1.0 / precision) + 0.5);
        return (double)value * precision;
    }

    // Build the one-sided analytic spectrum (Hilbert transform in frequency domain)
    void construct_analytic_spectrum(int n_fft, const fftw_complex *half, fftw_complex *full)
    {
        const bool even = (n_fft % 2 == 0);
        const int nyquist = n_fft / 2;
        // Last bin that gets doubled: the Nyquist bin itself is real and stays unscaled for even n_fft.
        const int last_doubled = even ? nyquist - 1 : nyquist;

        full[0][0] = half[0][0];
        full[0][1] = half[0][1];

        for (int k = 1; k <= last_doubled; ++k)
        {
            full[k][0] = 2.0 * half[k][0];
            full[k][1] = 2.0 * half[k][1];
        }

        if (even)
        {
            full[nyquist][0] = half[nyquist][0];
            full[nyquist][1] = half[nyquist][1];
        }

        const int zero_start = last_doubled + 1 + (even ? 1 : 0);
        std::memset(full + zero_start, 0, (n_fft - zero_start) * sizeof(fftw_complex));
    }

} // namespace

PhaseEstimation::PhaseEstimation() : IProcessor(PRIORITY_HIGH)
{
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("calibrate", calibrate_, "Whether to apply MSE-optimal calibration gain to cecHT.");
    add_option("f0_read_interval", f0_read_interval_, "Packets between shared f0 polling steps.");
    add_option("filter", filter_def_, "Filter definition.", false);
    add_option("compensate_filter", compensate_filter_, "Whether to compensate the phase distortion of the preceding bandpass filter.", false);

    f0_state_ = create_follower_state<double>(
        "f0", std::numeric_limits<double>::quiet_NaN(), Permission::NONE,
        "Individual alpha frequency shared by an upstream processor.");
}

void PhaseEstimation::CreatePorts()
{
    data_in_port_ = create_input_port<MultiChannelType<double>>(
        "in",
        MultiChannelType<double>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
        PortInPolicy(SlotRange(0, MAX_NCHANNELS)));

    // Instantaneous phase (rad)
    phase_out_port_ = create_output_port<MultiChannelType<double>>(
        "phase",
        MultiChannelType<double>::Parameters(1, 1, 1), // Placeholder, set in CompleteStreamInfo
        PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));

    // Filtered analytic real part
    real_out_port_ = create_output_port<MultiChannelType<double>>(
        "real",
        MultiChannelType<double>::Parameters(1, 1, 1), // Placeholder, set in CompleteStreamInfo
        PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));
}

void PhaseEstimation::CompleteStreamInfo()
{
    const auto &input_params = data_in_port_->slot(0)->streaminfo().parameters<MultiChannelType<double>::Parameters>();
    const auto input_rate = data_in_port_->streaminfo(0).stream_rate();

    for (auto *port : {phase_out_port_, real_out_port_})
    {
        port->streaminfo(0).set_parameters(input_params);
        port->streaminfo(0).set_stream_rate(input_rate);
    }
}

void PhaseEstimation::load_filter_coeffs(const StorageContext &context, double f0)
{
    if (!filter_def_()["file"])
    {
        int N = 1;
        // double bandwidth = filter_def_()["bandwidth"].as<double>(4.0);
        double bandwidth = 0.9 * f0; // 90% of f0 as bandwidth
        double low_cutoff = f0 - bandwidth / 2.0;
        double high_cutoff = f0 + bandwidth / 2.0;
        int window_size = n_fft_;
        // Nudge away from exact .xx5 boundaries before rounding to 2 decimals, so that
        // tiny floating-point noise relative to the Python precompute script can't flip the rounding direction and produce
        // a mismatched filename.
        constexpr double tie_break_epsilon = 1e-9;
        std::string filename;
        filename = std::to_string(N) + "_" + std::format("{:.2f}", low_cutoff + tie_break_epsilon) + "_" + std::format("{:.2f}", high_cutoff + tie_break_epsilon) + "_" + std::to_string(fs_) + "_" + std::to_string(window_size) + ".txt";

        coeff_file_ = context.resolve_path(filename, "filters");
    }
    else
    {
        coeff_file_ = context.resolve_path(
            filter_def_()["file"].as<std::string>(), "filters");
    }

    coeffs_.clear();
    coeffs_.reserve(static_cast<size_t>(n_fft_));

    std::ifstream stream(coeff_file_);
    if (!stream.good())
    {
        throw std::runtime_error("PhaseEstimation: Cannot open filter coefficients file: " + coeff_file_);
    }

    auto header = dsp::filter::parse_file_header(stream);
    if (header["type"] != "frequency response")
    {
        throw std::runtime_error("PhaseEstimation: Expected frequency response in file");
    }

    if (calibrate_())
    {
        auto real_it = header.find("calibration gain real");
        auto imag_it = header.find("calibration gain imag");
        if (real_it != header.end() && imag_it != header.end())
        {
            c_gain_ = std::complex<double>(std::stod(real_it->second), std::stod(imag_it->second));
        }
        else
        {
            LOG(WARNING) << name() << " Calibration requested but " << coeff_file_
                         << " has no precomputed calibration gain; falling back to unity gain.";
            c_gain_ = std::complex<double>(1.0, 0.0);
        }
    }
    else
    {
        c_gain_ = std::complex<double>(1.0, 0.0);
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
            "PhaseEstimation: bandpass coefficient file has " + std::to_string(coeffs_.size()) + " bins, expected " + std::to_string(n_fft_));
    }
}

void PhaseEstimation::load_phase_shift(const StorageContext &context, double f0)
{
    std::string filename = "ecHTfilter_" + std::to_string(fs_) + "_phase.txt";
    std::string phase_shift_file_ = context.resolve_path(filename, "filters");

    filter_phase_shift_values_.clear();

    std::ifstream stream(phase_shift_file_);
    if (!stream.good())
    {
        throw std::runtime_error("PhaseEstimation: Cannot open phase shift file: " + phase_shift_file_);
    }

    auto header = dsp::filter::parse_file_header(stream);
    if (header["type"] != "phase shift")
    {
        throw std::runtime_error("PhaseEstimation: Expected phase shift in file:" + phase_shift_file_);
    }

    double phase_shift_rad;
    while (stream >> phase_shift_rad)
    {
        filter_phase_shift_values_.emplace_back(phase_shift_rad);
    }
    // Phase shift values are stored at 0.1 Hz increments, so index = round(f0 * 10)
    filter_phase_shift_ = filter_phase_shift_values_[static_cast<size_t>(std::round(f0 * 10))];
    LOG(INFO) << name() << " Loaded phase shift of " << filter_phase_shift_ << " radians for compensation";
}

void PhaseEstimation::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);
    const auto &p = info.parameters<MultiChannelType<double>::Parameters>();
    LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate;
    fs_ = p.sample_rate;

    window_size_ = static_cast<int>(fs_ * (1.0 / f0_) * 2.0); // 2 cycles of the current f0
    n_fft_ = static_cast<int>(good_size_real(window_size_));
    // Pre-size the circular buffer for the worst-case f0 (5 Hz → 2 cycles = 2/5 * fs samples)
    sample_window.set_capacity(static_cast<int>(fs_ * (1.0 / 5.0) * 2.0));
    LOG(INFO) << name() << " Sample window size set to " << window_size_ << ", FFT size: " << n_fft_;

    signal_in = fftw_alloc_real(n_fft_);
    freq_half = fftw_alloc_complex(n_fft_ / 2 + 1);
    freq = fftw_alloc_complex(n_fft_);
    out = fftw_alloc_complex(n_fft_);
    
    load_filter_coeffs(context, f0_);

    {
        std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
        fftw_import_wisdom_from_filename((context.resolve_path("fftw_wisdom.txt", "fft_wisdom")).c_str());

        p_ = fftw_plan_dft_r2c_1d(n_fft_, signal_in, freq_half, FFTW_WISDOM_ONLY);
        if (p_ == nullptr)
        {
            LOG(WARNING) << name() << " No wisdom available for FFT planning, using patient mode.";
            p_ = fftw_plan_dft_r2c_1d(n_fft_, signal_in, freq_half, FFTW_PATIENT);
        }
        p_inv_ = fftw_plan_dft_1d(n_fft_, freq, out, FFTW_BACKWARD, FFTW_WISDOM_ONLY);
        if (p_inv_ == nullptr)
        {
            LOG(WARNING) << name() << " No wisdom available for IFFT planning, using patient mode.";
            p_inv_ = fftw_plan_dft_1d(n_fft_, freq, out, FFTW_BACKWARD, FFTW_PATIENT);
        }
    }
}

void PhaseEstimation::Preprocess(ProcessingContext &context)
{
    packet_count_ = 0;

    f0_ = 10.0; // default f0, will be updated from shared state

    window_size_ = static_cast<int>(fs_ * (1.0 / f0_) * 2.0); // 2 cycles of the current f0
    n_fft_ = static_cast<int>(good_size_real(window_size_));
    // Pre-size the circular buffer for the worst-case f0 (5 Hz → 2 cycles = 2/5 * fs samples)
    sample_window.set_capacity(static_cast<int>(fs_ * (1.0 / 5.0) * 2.0));
    LOG(INFO) << name() << " Sample window size set to " << window_size_ << ", FFT size: " << n_fft_;

    load_filter_coeffs(context, f0_);

    if (compensate_filter_())
    {
        load_phase_shift(context, f0_);
    }

}

void PhaseEstimation::Process(ProcessingContext &context)
{
    MultiChannelType<double>::Data *data_in;
    MultiChannelType<double>::Data *data_real_out = nullptr;
    MultiChannelType<double>::Data *data_phase_out = nullptr;

    double sample;
    double phase;
    double real_part;

    while (!context.terminated())
    {
        if (n_messages_() != -1 && packet_count_ >= static_cast<size_t>(n_messages_()))
        {
            break;
        }

        if (!data_in_port_->slot(0)->RetrieveData(data_in))
        {
            break;
        }

        sample = data_in->data_sample(0, 0);

        data_phase_out = phase_out_port_->slot(0)->ClaimData(false);
        data_phase_out->set_hardware_timestamp(data_in->hardware_timestamp());

        data_real_out = real_out_port_->slot(0)->ClaimData(false);
        data_real_out->set_hardware_timestamp(data_in->hardware_timestamp());

        sample_window.push_back(sample);

        data_in_port_->slot(0)->ReleaseData();

        if (packet_count_ % f0_read_interval_() == 0)
        {
            double new_f0 = f0_state_->get();
            if (std::isnan(new_f0) || new_f0 < 5.0 || new_f0 > 18.0)
            {
                valid_f0_ = false;
            }
            else
            {
                valid_f0_ = true;
                // Round to 0.1 Hz to avoid excessive recalibration from small f0 fluctuations
                new_f0 = round(new_f0, 0.1);

                if (std::abs(new_f0 - f0_) >= 0.1)
                {
                    // f0 changed — update window, FFT, and calibration
                    f0_ = new_f0;
                    int old_n_fft = n_fft_;
                    window_size_ = static_cast<int>(2.0 * fs_ / f0_); // 2 cycles of new f0
                    if (window_size_ > sample_window.capacity())
                    {
                        LOG(WARNING) << name() << " New window size " << window_size_ << " exceeds circular buffer capacity " << sample_window.capacity() << ". Resizing.";
                        sample_window.rset_capacity(window_size_);
                    }
                    n_fft_ = good_size_real(window_size_);
                    LOG(DEBUG) << name() << " Packet " << packet_count_ << ": f0 updated to " << f0_ << " Hz, window: " << window_size_ << ", FFT size: " << n_fft_;

                    load_filter_coeffs(context, f0_);

                    if (old_n_fft != n_fft_)
                    {
                        // Reallocate FFTW arrays for the new FFT size
                        {
                            std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
                            fftw_destroy_plan(p_);
                            fftw_destroy_plan(p_inv_);
                        }
                        fftw_free(signal_in);  signal_in = nullptr;
                        fftw_free(freq_half);  freq_half = nullptr;
                        fftw_free(freq);       freq      = nullptr;
                        fftw_free(out);        out       = nullptr;

                        signal_in = fftw_alloc_real(n_fft_);
                        freq_half = fftw_alloc_complex(n_fft_ / 2 + 1);
                        freq = fftw_alloc_complex(n_fft_);
                        out = fftw_alloc_complex(n_fft_);
                        {
                            std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
                            p_ = fftw_plan_dft_r2c_1d(n_fft_, signal_in, freq_half, FFTW_WISDOM_ONLY);
                            if (p_ == nullptr)
                            {
                                LOG(WARNING) << name() << " No wisdom for FFT (" << n_fft_ << "), using estimate mode.";
                                p_ = fftw_plan_dft_r2c_1d(n_fft_, signal_in, freq_half, FFTW_ESTIMATE);
                            }
                            p_inv_ = fftw_plan_dft_1d(n_fft_, freq, out, FFTW_BACKWARD, FFTW_WISDOM_ONLY);
                            if (p_inv_ == nullptr)
                            {
                                LOG(WARNING) << name() << " No wisdom for IFFT (" << n_fft_ << "), using estimate mode.";
                                p_inv_ = fftw_plan_dft_1d(n_fft_, freq, out, FFTW_BACKWARD, FFTW_ESTIMATE);
                            }
                        }
                    }

                    if (compensate_filter_())
                    {
                        // Phase shift values are stored at 0.1 Hz increments
                        filter_phase_shift_ = filter_phase_shift_values_[static_cast<size_t>(std::round(f0_ * 10))];
                    }
                }
            }
        }

        if (valid_f0_ && (sample_window.size() >= static_cast<size_t>(window_size_)))
        {
            // Copy the most recent window_size_ samples into contiguous FFTW input; zero-pad
            auto a1 = sample_window.array_one();
            auto a2 = sample_window.array_two();
            size_t skip = sample_window.size() - static_cast<size_t>(window_size_);
            if (skip >= a1.second)
            {
                // Entire window lies within the second (wrapped) chunk
                std::copy(a2.first + (skip - a1.second), a2.first + a2.second, signal_in);
            }
            else
            {
                // Window spans the boundary between the two chunks
                size_t from_one = a1.second - skip;
                std::copy(a1.first + skip, a1.first + a1.second, signal_in);
                std::copy(a2.first, a2.first + a2.second, signal_in + from_one);
            }
            std::fill(signal_in + window_size_, signal_in + n_fft_, 0.0);

            fftw_execute(p_);

            construct_analytic_spectrum(n_fft_, freq_half, freq);

            // Multiply analytic spectrum with bandpass filter coefficients
            for (int k = 0; k < n_fft_; k++)
            {
                const double in_re = freq[k][0];
                const double in_im = freq[k][1];
                const double c_re = coeffs_[k].real();
                const double c_im = coeffs_[k].imag();
                freq[k][0] = in_re * c_re - in_im * c_im;
                freq[k][1] = in_re * c_im + in_im * c_re;
            }

            fftw_execute(p_inv_);

            // Normalize IFFT output and apply calibration gain
            double c_gain_re = c_gain_.real();
            double c_gain_im = c_gain_.imag();
            for (int i = 0; i < n_fft_; i++)
            {
                const double in_re = out[i][0];
                const double in_im = out[i][1];
                out[i][0] = (in_re * c_gain_re - in_im * c_gain_im) / n_fft_; // real
                out[i][1] = (in_re * c_gain_im + in_im * c_gain_re) / n_fft_; // imag
            }

            // Extract phase and real part at the last sample of the window
            phase = std::atan2(out[window_size_ - 1][1], out[window_size_ - 1][0]);
            // Compensate for the phase distortion introduced by the preceding bandpass filter
            phase = std::atan2(std::sin(phase - filter_phase_shift_), std::cos(phase - filter_phase_shift_));
            real_part = out[window_size_ - 1][0];

            data_phase_out->set_data_sample(0, 0, phase);
            data_phase_out->set_source_timestamp(Clock::now());
            data_phase_out->set_sample_timestamps(data_in->sample_timestamps());
            data_real_out->set_data_sample(0, 0, real_part);
            data_real_out->set_source_timestamp(Clock::now());
            data_real_out->set_sample_timestamps(data_in->sample_timestamps());
        }
        else
        {
            // Output NaN while f0 is invalid or window is not yet full
            data_phase_out->set_data_sample(0, 0, std::numeric_limits<double>::quiet_NaN());
            data_phase_out->set_source_timestamp(Clock::now());
            data_phase_out->set_sample_timestamps(data_in->sample_timestamps());
            data_real_out->set_data_sample(0, 0, std::numeric_limits<double>::quiet_NaN());
            data_real_out->set_source_timestamp(Clock::now());
            data_real_out->set_sample_timestamps(data_in->sample_timestamps());
        }

        phase_out_port_->slot(0)->PublishData();
        real_out_port_->slot(0)->PublishData();

        packet_count_++;
    }
    LOG(INFO) << name() << " stopped working";
}

void PhaseEstimation::Postprocess(ProcessingContext &context)
{
    LOG(INFO) << name() << ": Total messages processed: " << packet_count_;
}

void PhaseEstimation::Unprepare(GlobalContext &context)
{
    {
        std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
        int ret = fftw_export_wisdom_to_filename(context.resolve_path("fftw_wisdom.txt", "fft_wisdom").c_str());
        if (ret == 0)
        {
            LOG(WARNING) << name() << " Failed to save FFTW wisdom to file.";
        }
        fftw_destroy_plan(p_);
        fftw_destroy_plan(p_inv_);
    }
    if (signal_in) { fftw_free(signal_in); signal_in = nullptr; }
    if (freq_half) { fftw_free(freq_half); freq_half = nullptr; }
    if (freq) { fftw_free(freq); freq = nullptr; }
    if (out) { fftw_free(out); out = nullptr; }
}

REGISTERPROCESSOR(PhaseEstimation);
