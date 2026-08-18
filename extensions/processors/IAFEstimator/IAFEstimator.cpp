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
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <sstream>
#include <string>
#include <fftw3.h>
#include <complex>
#include <dsp/fftw_planner_mutex.hpp>

namespace
{
    struct PeakSeed
    {
        int bin = -1;
        double iaf_hz = std::numeric_limits<double>::quiet_NaN();
        double peak_value = std::numeric_limits<double>::quiet_NaN();
    };

    struct PeakFitResult
    {
        bool valid = false;
        int bin = -1;
        double iaf_hz = std::numeric_limits<double>::quiet_NaN();
        double sigma_hz = std::numeric_limits<double>::quiet_NaN();
        double amplitude = std::numeric_limits<double>::quiet_NaN();
        double delta_bic = std::numeric_limits<double>::quiet_NaN();
    };

    // Returns the next FFT-efficient size >= n (favors factors of 2, 3, 5).
    // From https://github.com/hayguen/pocketfft/blob/cpp/pocketfft_hdronly.h
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

    std::vector<double> gaussian(const std::vector<double> &freqs, double amplitude, double center, double width)
    {
        std::vector<double> gauss(freqs.size());
        for (size_t k = 0; k < freqs.size(); ++k)
        {
            gauss[k] = amplitude * std::exp(-0.5 * std::pow((freqs[k] - center) / width, 2));
        }
        return gauss;
    }

    PeakSeed find_peak_seed(const std::vector<double> &smoothed_power, int f_min_bin, int f_max_bin, double resolution, bool parabolic = true)
    {
        PeakSeed seed;

        int max_bin = -1;
        double max_value = -1.0;
        double iaf = -1.0;

        for (size_t k = f_min_bin; k <= f_max_bin; ++k)
        {
            if (smoothed_power[k] > max_value)
            {
                max_value = smoothed_power[k];
                max_bin = static_cast<int>(k);
                iaf = max_bin * resolution;
            }
        }

        if (max_bin < 0)
        {
            return seed;
        }

        // Sub-bin refinement via parabolic interpolation
        if (parabolic && max_bin > f_min_bin && max_bin < f_max_bin)
        {
            double y1 = smoothed_power[max_bin - 1];
            double y2 = smoothed_power[max_bin];
            double y3 = smoothed_power[max_bin + 1];
            double denom = (y1 - 2 * y2 + y3);
            if (denom != 0)
            {
                double delta = 0.5 * (y1 - y3) / denom;
                iaf += delta * resolution;
            }
        }

        seed.bin = max_bin;
        seed.iaf_hz = iaf;
        seed.peak_value = max_value;
        return seed;
    }

    PeakFitResult fit_gaussian_peak(const std::vector<double> &smoothed_power, const PeakSeed &seed, double resolution)
    {
        PeakFitResult result;

        double amplitude = seed.peak_value;
        if (amplitude <= 0.0)
        {
            return result;
        }

        // FWHM-based sigma estimate
        double half_max = amplitude / 2.0;
        int left_half_bin = seed.bin;
        while (left_half_bin > 0 && smoothed_power[left_half_bin] > half_max)
        {
            --left_half_bin;
        }

        int right_half_bin = seed.bin;
        while (right_half_bin < smoothed_power.size() - 1 && smoothed_power[right_half_bin] > half_max)
        {
            ++right_half_bin;
        }

        double fwhm = std::max(resolution, (right_half_bin - left_half_bin) * resolution);
        double std_gauss = fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0)));

        result.bin = seed.bin;
        result.iaf_hz = seed.iaf_hz;
        result.sigma_hz = std_gauss;
        result.amplitude = amplitude;

        return result;
    }

    bool bic_test(const std::vector<double> &power, const std::vector<double> &freqs, const std::vector<double> &gauss, PeakFitResult &peak, int f_min_bin, int f_max_bin)
    {
        size_t n = static_cast<size_t>(f_max_bin - f_min_bin + 1);
        double ss_h0 = 0.0;
        double ss_h1 = 0.0;
        for (int k = f_min_bin; k <= f_max_bin; ++k)
        {
            double centered = power[k];
            ss_h0 += (centered * centered);
            double resid = power[k] - gauss[k];
            ss_h1 += (resid * resid);
        }

        double bic_null = n * std::log(ss_h0 / n);
        double bic_peak = n * std::log(ss_h1 / n) + 3.0 * std::log(static_cast<double>(n));
        double delta_bic = bic_null - bic_peak;
        peak.delta_bic = delta_bic;
        return delta_bic > 0.0;
    }

    struct LinearFitResult
    {
        double slope;
        double intercept;
        double r_squared;
    };

    LinearFitResult linear_regression(const std::vector<double> &x, const std::vector<double> &y)
    {
        size_t n = x.size();
        double x_mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
        double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / n;

        double ss_xy = 0.0, ss_xx = 0.0, ss_yy = 0.0;
        for (int i = 0; i < n; i++)
        {
            ss_xy += (x[i] - x_mean) * (y[i] - y_mean);
            ss_xx += (x[i] - x_mean) * (x[i] - x_mean);
            ss_yy += (y[i] - y_mean) * (y[i] - y_mean);
        }

        double slope = ss_xy / ss_xx;
        double intercept = y_mean - slope * x_mean;
        double r_squared = (ss_xy * ss_xy) / (ss_xx * ss_yy);

        return {slope, intercept, r_squared};
    }

    void power_log(std::vector<double> &power)
    {
        // Clamp to smallest positive value so log operations never see zero
        for (double &p : power)
        {
            p = std::log10(std::max(p, std::numeric_limits<double>::denorm_min()));
        }
    }

    std::vector<double> get_aperiodic(const std::vector<double> &log_power, const std::vector<double> &freqs)
    {
        size_t N = log_power.size();

        // Ignore DC bin
        std::vector<double> log_freqs(N - 1);
        std::vector<double> log_power_noDC(N - 1);
        for (size_t k = 1; k < N; ++k)
        {
            log_freqs[k - 1] = std::log10(freqs[k]);
            log_power_noDC[k - 1] = log_power[k];
        }

        LinearFitResult fit = linear_regression(log_freqs, log_power_noDC);

        std::vector<double> power_flat(N-1);
        std::vector<double> aperiodic(N);
        for (size_t k = 0; k < N - 1; ++k)
        {
            double aperiodic_fit = fit.slope * log_freqs[k] + fit.intercept;
            power_flat[k] = log_power_noDC[k] - aperiodic_fit;
        }

        std::vector<double> log_freqs_select;
        std::vector<double> log_power_select;
        log_freqs_select.reserve(N - 1);
        log_power_select.reserve(N - 1);
        for (size_t k = 0; k < N - 1; ++k)
        {
            // Exclude bins above the aperiodic fit to avoid bias from oscillatory peaks
            if (power_flat[k] <= 0.0)
            {
                log_freqs_select.push_back(log_freqs[k]);
                log_power_select.push_back(log_power_noDC[k]);
            }
        }
        fit = linear_regression(log_freqs_select, log_power_select);
        for (size_t k = 1; k < N; ++k)
        {
            aperiodic[k] = fit.slope * log_freqs[k-1] + fit.intercept;
        }
        aperiodic[0] = log_power[0]; // DC bin is not used in the fit, so just copy the original value to subtract it perfectly

        return aperiodic;
    }

    void power_flat(const std::vector<double> &aperiodic, std::vector<double> &log_power)
    {
        size_t N = log_power.size();
        for (size_t k = 0; k < N; ++k)
        {
            log_power[k] -= aperiodic[k];
        }
    }

    std::vector<double> savgol_filter(const std::vector<double> &weights, const std::vector<double> &power)
    {
        std::vector<double> filtered(power.size());

        int window_size = weights.size();
        int half_window_size = window_size / 2;

        // Pad with nearest value (mode=nearest)
        std::vector<double> data = power;
        data.insert(data.begin(), half_window_size, data.front());
        data.insert(data.end(), half_window_size, data.back());

        for (int i = 0; i < power.size(); i++)
        {
            double res = 0.0;
            for (int j = 0; j < window_size; ++j)
            {
                res += weights[j] * data[i + j];
            }
            filtered[i] = res;
        }

        return filtered;
    }

} // namespace

IAFEstimator::IAFEstimator() : IProcessor(PRIORITY_HIGH)
{
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("window_size_sec", window_size_sec_, "Window size in seconds.");
    add_option("f_min", f_min_, "Left bound of alpha search range (Hz).");
    add_option("f_max", f_max_, "Right bound of alpha search range (Hz).");
    add_option("calc_interval", calc_interval_, "Number of packets between IAF calculations.");
    add_option("max_invalid_sec", max_invalid_sec_, "Maximum duration of invalid data in seconds before reset of estimation.");
    add_option("kalman_iaf_std", kalman_iaf_std_, "Std of the IAF drift [Hz/s]. Sets Kalman Q.");
    add_option("kalman_full", kalman_full_, "If true, use full Kalman filter with adaptive gain and cold-start. If false, use EMA-equivalent fixed gain.");

    iaf_state_ = create_broadcaster_state<double>(
        "iaf", current_iaf_, Permission::NONE,
        "Individual alpha frequency shared with downstream processors.");
}

void IAFEstimator::CreatePorts()
{
    data_in_port_ = create_input_port<MultiChannelType<double>>(
        "in",
        MultiChannelType<double>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
        PortInPolicy(SlotRange(0, MAX_NCHANNELS)));

    data_out_port_ = create_output_port<ScalarType<double>>(
        "out",
        ScalarType<double>::Parameters(1), // Placeholder, set in CompleteStreamInfo
        PortOutPolicy(SlotRange(0, MAX_NCHANNELS), 200, WaitStrategy::kBlockingStrategy));
}

void IAFEstimator::CompleteStreamInfo()
{
    for (int k = 0; k < data_out_port_->number_of_slots(); ++k)
    {
        data_out_port_->streaminfo(k).set_stream_rate(data_in_port_->streaminfo(0).stream_rate() / calc_interval_());
    }
}

void IAFEstimator::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);
    const auto &p = info.parameters<MultiChannelType<double>::Parameters>();
    LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate;
    fs_ = p.sample_rate;
    window_size_ = window_size_sec_() * fs_;
    n_fft_ = static_cast<int>(good_size_real(window_size_));
    sample_window.set_capacity(n_fft_);
    LOG(INFO) << name() << " Sample window size set to " << window_size_ << ", FFT size: " << n_fft_;

    iaf_state_->set(current_iaf_);

    const double update_interval_s = static_cast<double>(calc_interval_()) / p.sample_rate;

    kf_R_ = 0.9933; // measurement noise variance (empirically determined)

    const double drift_var_per_s = kalman_iaf_std_() * kalman_iaf_std_();
    kf_Q_ = drift_var_per_s * update_interval_s;

    // Full KF: start uncertain (P = R) for fast cold-start acquisition.
    // EMA mode: start at steady-state (P = sqrt(Q*R)) so gain is fixed from sample 0.
    kf_P_ = kalman_full_() ? kf_R_ : std::sqrt(kf_Q_ * kf_R_);

    const double K_steady = (-kf_Q_ + std::sqrt(kf_Q_ * kf_Q_ + 4.0 * kf_Q_ * kf_R_)) / (2.0 * kf_R_);
    const double alpha_equivalent = 1.0 - K_steady;
    const double tau_equivalent = -update_interval_s / std::log(alpha_equivalent);

    invalid_threshold_ = static_cast<int>((fs_ * max_invalid_sec_()) / calc_interval_());

    LOG(INFO) << name() << " Kalman filter configured:"
              << " Q=" << kf_Q_ << " R=" << kf_R_
              << " P0=" << kf_P_
              << " K_steady=" << K_steady
              << " alpha_equivalent=" << alpha_equivalent
              << " tau_equivalent=" << tau_equivalent << "s"
              << " mode=" << (kalman_full_() ? "full KF" : "EMA-equivalent")
              << " invalid_threshold=" << invalid_threshold_ << " estimates";

    freq_resolution_ = fs_ / n_fft_;
    LOG(INFO) << name() << " Frequency resolution: " << freq_resolution_ << " Hz";

    int savgol_window_length = static_cast<int>(2.5 / freq_resolution_); // 2.5 Hz smoothing window
    if (savgol_window_length % 2 == 0)
    {
        savgol_window_length += 1; // must be odd
    }
    int savgol_polyorder = 3;
    if (savgol_polyorder >= savgol_window_length)
    {
        savgol_window_length = savgol_polyorder + 2;
    }
    int m = savgol_window_length / 2;
    LOG(INFO) << name() << " Savitzky-Golay filter length: " << savgol_window_length << ", polynomial order: " << savgol_polyorder << ", m: " << m;
    savgol_weights_ = gram_sg::compute_weights(m, 0, savgol_polyorder, 0);

    f_min_bin_ = std::floor(f_min_() / freq_resolution_);
    f_max_bin_ = std::ceil(f_max_() / freq_resolution_);
    LOG(INFO) << name() << " IAF search range: " << f_min_() << " - " << f_max_() << " Hz (bins " << f_min_bin_ << " - " << f_max_bin_ << ")";

    max_analyze_bin_ = 30 / freq_resolution_ + 1; // analyze up to 30 Hz to avoid high-frequency noise

    signal_in = fftw_alloc_real(n_fft_);
    freq_half = fftw_alloc_complex(n_fft_ / 2 + 1);

    freqs_.resize(max_analyze_bin_);
    for (size_t k = 0; k < max_analyze_bin_; ++k)
    {
        freqs_[k] = static_cast<double>(k) * freq_resolution_;
    }

    {
        fftw_import_wisdom_from_filename(context.resolve_path("fftw_wisdom.txt", "fft_wisdom").c_str());
        std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
        fft_plan_ = fftw_plan_dft_r2c_1d(n_fft_, signal_in, freq_half, FFTW_WISDOM_ONLY);
        if (fft_plan_ == nullptr)
        {
            LOG(WARNING) << name() << " No wisdom available for FFT planning, using patient mode.";
            fft_plan_ = fftw_plan_dft_r2c_1d(n_fft_, signal_in, freq_half, FFTW_PATIENT);
        }
    }
}

void IAFEstimator::Preprocess(ProcessingContext &context)
{
    current_iaf_ = std::numeric_limits<double>::quiet_NaN();
    iaf_state_->set(current_iaf_);
    last_valid_iaf_ = std::numeric_limits<double>::quiet_NaN();
    current_gauss_width_ = std::numeric_limits<double>::quiet_NaN();
    kf_x_ = std::numeric_limits<double>::quiet_NaN();
    kf_P_ = kalman_full_() ? kf_R_ : std::sqrt(kf_Q_ * kf_R_);
    packet_count_ = 0;
    invalid_count_ = 0;
}

void IAFEstimator::Process(ProcessingContext &context)
{
    MultiChannelType<double>::Data *data_in;
    ScalarType<double>::Data *data_out;

    auto kalman_predict = [&]() {
        kf_P_ = kf_P_ + kf_Q_;
    };

    auto kalman_update = [&](double measurement, double R_n) {
        if (kalman_full_())
        {
            kf_R_ = R_n;
        }
        const double K = kf_P_ / (kf_P_ + kf_R_);
        kf_x_ = kf_x_ + K * (measurement - kf_x_);
        kf_P_ = (1.0 - K) * kf_P_;
    };

    while (!context.terminated())
    {
        if (n_messages_() != -1 && packet_count_ >= n_messages_())
        {
            break;
        }

        if (!data_in_port_->slot(0)->RetrieveData(data_in))
        {
            break;
        }
        TimePoint start_time = Clock::now();

        data_out = data_out_port_->slot(0)->ClaimData(false);

        sample_window.push_back(data_in->data_sample(0, 0));
        data_out->set_hardware_timestamp(data_in->hardware_timestamp());

        data_in_port_->slot(0)->ReleaseData();

        if ((sample_window.size() == sample_window.capacity()) && (packet_count_ % calc_interval_() == 0))
        {
            // Copy circular buffer into contiguous FFTW input array; zero-pad to n_fft_
            for (int i = 0; i < window_size_; ++i)
            {
                signal_in[i] = sample_window[i];
            }
            for (int i = window_size_; i < n_fft_; ++i)
            {
                signal_in[i] = 0.0;
            }

            fftw_execute(fft_plan_);

            std::vector<double> power(max_analyze_bin_);
            for (size_t k = 0; k < max_analyze_bin_; ++k)
            {
                power[k] = (pow(freq_half[k][0], 2) + pow(freq_half[k][1], 2)) / (n_fft_ * fs_);
            }
            power_log(power);

            std::vector<double> aperiodic = get_aperiodic(power, freqs_);

            power_flat(aperiodic, power);

            std::vector<double> power_smooth(max_analyze_bin_);
            power_smooth = savgol_filter(savgol_weights_, power);

            PeakSeed seed = find_peak_seed(power_smooth, f_min_bin_, f_max_bin_, freq_resolution_, true);
            PeakFitResult peak = fit_gaussian_peak(power_smooth, seed, freq_resolution_);

            std::vector<double> gauss = gaussian(freqs_, peak.amplitude, peak.iaf_hz, peak.sigma_hz);

            if (peak.sigma_hz > max_gauss_width_hz_())
            {
                peak.valid = false; // reject peaks that are too broad
            }
            else
            {
                peak.valid = bic_test(power, freqs_, gauss, peak, 0, freqs_.size() - 1);
            }

            // Kalman predict: uncertainty grows between updates
            kalman_predict();

            if (peak.valid)
            {
                invalid_count_ = std::max(0, invalid_count_ - 1);
                current_iaf_ = peak.iaf_hz;
                last_valid_iaf_ = peak.iaf_hz;
                current_gauss_width_ = peak.sigma_hz;
                if (std::isnan(kf_x_))
                {
                    // First valid estimate — initialize directly (no smoothing yet)
                    kf_x_ = peak.iaf_hz;
                }
                else
                {
                    if (kalman_full_())
                    {
                        double signal_power = 0.0;
                        double noise_power = 0.0;
                        // Compute SNR from Gaussian peak power within ±2σ
                        for (int k = 0; k < max_analyze_bin_; ++k)
                        {
                            if (std::abs(freqs_[k] - peak.iaf_hz) <= 2 * peak.sigma_hz)
                            {
                                double aperiodic_lin = std::pow(10, aperiodic[k]);
                                double gauss_lin = std::pow(10, gauss[k]) - 1.0;
                                signal_power += aperiodic_lin * gauss_lin;
                                noise_power += aperiodic_lin;
                            }
                        }
                        SNR_ = signal_power / std::max(noise_power, std::numeric_limits<double>::denorm_min());

                        double R_n = current_gauss_width_ * current_gauss_width_ / (2 * SNR_);
                        kalman_update(peak.iaf_hz, R_n);
                    }
                    else
                    {
                        // EMA-equivalent fixed-gain update
                        kalman_update(peak.iaf_hz, kf_R_);
                    }
                }
            }
            else
            {
                invalid_count_ = std::min(invalid_count_ + 1, invalid_threshold_ + 1);
                current_iaf_ = std::numeric_limits<double>::quiet_NaN();
                current_gauss_width_ = std::numeric_limits<double>::quiet_NaN();
            }
            if (invalid_count_ == invalid_threshold_)
            {
                LOG(WARNING) << name() << " Packet " << packet_count_ << ": Too many consecutive invalid estimates, resetting IAF estimation.";
            }
            if (invalid_count_ >= invalid_threshold_)
            {
                kf_x_ = std::numeric_limits<double>::quiet_NaN();
                kf_P_ = kalman_full_() ? kf_R_ : std::sqrt(kf_Q_ * kf_R_);
            }

            iaf_state_->set(kf_x_);
            TimePoint end_time = Clock::now();
            if (packet_count_ % static_cast<int>(fs_) == 0)
            {
                LOG(INFO) << name() << " Packet " << packet_count_ << " (" << invalid_count_ << " invalid): Estimated IAF = " << current_iaf_ << " Hz (sigma: " << current_gauss_width_ << "), KF estimate: " << kf_x_ << " Hz (R=" << kf_R_ << ", SNR(dB)=" << 20 * std::log10(SNR_) << "), took " << std::chrono::duration<double, std::micro>(end_time - start_time).count() << " us";
            }
        }
        data_out->set_data(kf_x_);
        data_out->set_source_timestamp(Clock::now());
        data_out_port_->slot(0)->PublishData();

        packet_count_++;
    }

    LOG(INFO) << name() << " stopped working";
}

void IAFEstimator::Postprocess(ProcessingContext &context)
{
    sample_window.clear();
    iaf_state_->set(std::numeric_limits<double>::quiet_NaN());
    LOG(INFO) << name() << ": Total messages processed: " << packet_count_;
}

void IAFEstimator::Unprepare(GlobalContext &context)
{
    {
        std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
        int ret = fftw_export_wisdom_to_filename(context.resolve_path("fftw_wisdom.txt", "fft_wisdom").c_str());
        if (ret == 0)
        {
            LOG(WARNING) << name() << " Failed to save FFTW wisdom.";
        }
        fftw_destroy_plan(fft_plan_);
    }
    fftw_free(signal_in);
    fftw_free(freq_half);
}

REGISTERPROCESSOR(IAFEstimator);
