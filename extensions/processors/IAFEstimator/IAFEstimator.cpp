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

    // int get_max_bin(fftwf_complex *freq, size_t N)
    // {
    //     int max_bin = -1;
    //     double max_mag2 = -1.0;

    //     for (size_t k = 0; k < N; ++k)
    //     {
    //         double mag2 = pow(freq[k][0], 2) + pow(freq[k][1], 2);

    //         if (mag2 > max_mag2)
    //         {
    //             max_mag2 = mag2;
    //             max_bin = static_cast<int>(k);
    //         }
    //     }
    //     return max_bin;
    // }

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
            // Decide between greater or greater/equal based on whether we want first peak or last peak in case of ties
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

    bool bic_test(const std::vector<double> &smoothed_power, const std::vector<double> &freqs, PeakFitResult &peak, int f_min_bin, int f_max_bin)
    {

        std::vector<double> gauss = gaussian(freqs, peak.amplitude, peak.iaf_hz, peak.sigma_hz);

        size_t n = static_cast<size_t>(f_max_bin - f_min_bin + 1);
        double ss_h0 = 0.0;
        double ss_h1 = 0.0;
        for (int k = f_min_bin; k <= f_max_bin; ++k)
        {
            double centered = smoothed_power[k];
            ss_h0 += centered * centered;
            double resid = smoothed_power[k] - gauss[k];
            ss_h1 += resid * resid;
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

    std::vector<double> remove_aperiodic(const std::vector<double> &power, const std::vector<double> &freqs)
    {
        size_t N = power.size();
        // Safe power (no zeros)
        std::vector<double> safe_power(N);
        for (size_t k = 0; k < N; ++k)
        {
            safe_power[k] = std::max(power[k], 1e-12);
        }

        // Ignore DC-bin
        std::vector<double> log_freqs(N - 1);
        std::vector<double> log_power(N - 1);
        for (size_t k = 1; k < N; ++k)
        {
            log_freqs[k - 1] = std::log10(freqs[k]);
            log_power[k - 1] = std::log10(safe_power[k]);
        }

        LinearFitResult fit = linear_regression(log_freqs, log_power);

        // LOG(INFO) << "Aperiodic fit: slope = " << fit.slope << ", intercept = " << fit.intercept << ", R^2 = " << fit.r_squared;

        std::vector<double> power_flat(N);

        for (size_t k = 1; k < N; ++k)
        {
            double aperiodic_fit = fit.slope * log_freqs[k - 1] + fit.intercept;
            power_flat[k] = safe_power[k] / std::pow(10, aperiodic_fit);
        }
        power_flat[0] = power_flat[1]; // Set DC bin to same as first non-DC bin to avoid causing artificial rise/fall
        std::vector<double> log_freqs_select;
        std::vector<double> log_power_select;

        log_freqs_select.reserve(N - 1);
        log_power_select.reserve(N - 1);
        for (size_t k = 1; k < N; ++k)
        {
            if (power_flat[k] <= 1.0) // Limit to perfect fit (=1) or below to avoid bias of oscillatory peaks above the fit
            {
                log_freqs_select.push_back(log_freqs[k - 1]);
                log_power_select.push_back(log_power[k - 1]);
            }
        }
        fit = linear_regression(log_freqs_select, log_power_select);
        for (size_t k = 1; k < N; ++k)
        {
            double aperiodic_fit = fit.slope * log_freqs[k - 1] + fit.intercept;
            power_flat[k] = safe_power[k] / std::pow(10, aperiodic_fit);
        }

        return power_flat;
    }

    std::vector<double> savgol_filter(gram_sg::SavitzkyGolayFilter savgol, const std::vector<double> &power)
    {
        std::vector<double> filtered(power.size());

        int window_size = savgol.config().window_size();
        int half_window_size = window_size / 2;

        std::vector<double> data = power;
        // Use mode=nearest for padding
        data.insert(data.begin(), half_window_size, data.front());
        data.insert(data.end(), half_window_size, data.back());

        std::vector<double> window;
        for (int i = 0; i < power.size(); i++)
        {
            for (int j = i; j < i + window_size; ++j)
            {
                window.push_back(data[j]);
            }
            filtered[i] = savgol.filter(window);
            window.clear();
        }

        return filtered;
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

} // namespace

IAFEstimator::IAFEstimator() : IProcessor(PRIORITY_HIGH)
{
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("window_size_sec", window_size_sec_, "Window size in seconds.");
    add_option("f_min", f_min_, "Left bound of alpha search range.");
    add_option("f_max", f_max_, "Right bound of alpha search range.");
    add_option("calc_interval", calc_interval_, "Number of packets between IAF calculations.");
    add_option("max_invalid_sec", max_invalid_sec, "Maximum duration of invalid data in seconds before reset of estimation.");
    add_option("kalman_estimator_std", kalman_estimator_std_, "Expected std of the IAF estimator output [Hz]. Sets Kalman R.");
    add_option("kalman_iaf_std", kalman_iaf_std_, "Std of the IAF drift [Hz/s]. Sets Kalman Q.");
    add_option("kalman_full", kalman_full_, "If true, use full Kalman filter with adaptive gain and cold-start. If false, use EMA-equivalent fixed gain.");

    iaf_state_ = create_broadcaster_state<double>(
        "iaf", current_iaf_, Permission::NONE,
        "Individual alpha frequency shared with downstream processors.");
}

void IAFEstimator::CreatePorts()
{
    data_in_port_ = create_input_port<MultiChannelType<float>>(
        "in",
        MultiChannelType<float>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
        PortInPolicy(SlotRange(0, MAX_NCHANNELS)));

    data_out_port_ = create_output_port<ScalarType<double>>(
        "out",
        ScalarType<double>::Parameters(1), // Placeholder, will be set in CompleteStreamInfo
        PortOutPolicy(SlotRange(0, MAX_NCHANNELS), 200, WaitStrategy::kBlockingStrategy));
}

void IAFEstimator::CompleteStreamInfo()
{
    // Set the parameters for the output stream
    for (int k = 0; k < data_out_port_->number_of_slots(); ++k)
    {
        data_out_port_->streaminfo(k).set_stream_rate(data_in_port_->streaminfo(0).stream_rate() / calc_interval_());
    }
}

void IAFEstimator::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);
    const auto &p = info.parameters<MultiChannelType<float>::Parameters>();
    LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate;
    fs_ = p.sample_rate;
    window_size_ = window_size_sec_() * fs_; // Convert window size from seconds to samples
    n_fft_ = static_cast<int>(good_size_real(window_size_));
    sample_window.set_capacity(n_fft_); // Initialize window size with next fast fft len
    LOG(INFO) << name() << " Sample window size set to " << window_size_ << ", FFT size: " << n_fft_;

    iaf_state_->set(current_iaf_);

    const double update_interval_s = static_cast<double>(calc_interval_()) / p.sample_rate;

    // R: measurement noise variance from estimator std
    kf_R_ = kalman_estimator_std_() * kalman_estimator_std_();

    // Q: process noise variance per update step
    const double drift_var_per_s = kalman_iaf_std_() * kalman_iaf_std_();

    kf_Q_ = drift_var_per_s * update_interval_s;

    // P initial value:
    //   full KF  → start uncertain (P = R) for fast cold-start acquisition
    //   EMA mode → start at steady-state (P = sqrt(Q*R)) so gain is fixed from sample 0
    kf_P_ = kalman_full_() ? kf_R_ : std::sqrt(kf_Q_ * kf_R_);

    const double K_steady = (-kf_Q_ + std::sqrt(kf_Q_ * kf_Q_ + 4.0 * kf_Q_ * kf_R_)) / (2.0 * kf_R_);

    const double alpha_equivalent = 1.0 - K_steady;
    const double tau_equivalent   = -update_interval_s / std::log(alpha_equivalent);

    invalid_threshold_ = static_cast<int>(std::ceil(tau_equivalent * p.sample_rate / calc_interval_()));

    LOG(INFO) << name() << " Kalman filter configured:"
              << " Q=" << kf_Q_ << " R=" << kf_R_
              << " P0=" << kf_P_
              << " K_steady=" << K_steady
              << " alpha_equivalent=" << alpha_equivalent
              << " tau_equivalent=" << tau_equivalent << "s"
              << " mode=" << (kalman_full_() ? "full KF" : "EMA-equivalent")
              << " invalid_threshold=" << invalid_threshold_ << " estimates";

    // Load FFTW wisdom if available to speed up plan creation
    fftwf_import_wisdom_from_filename(context.resolve_path("fftw_wisdom.txt", "fft_wisdom").c_str());

    freq_resolution = fs_ / n_fft_;
    LOG(INFO) << name() << " Frequency resolution: " << freq_resolution << " Hz";

    int savgol_window_length = static_cast<int>(2.5 / freq_resolution); // 2.5 Hz window for smoothing
    if (savgol_window_length % 2 == 0)
    {
        savgol_window_length += 1; // Ensure window length is odd
    }

    int savgol_polyorder = 3;
    if (savgol_polyorder >= savgol_window_length)
    {
        savgol_window_length = savgol_polyorder + 2; // Ensure window length is greater than polynomial order
    }
    int m = (savgol_window_length) / 2;
    LOG(INFO) << name() << " Savitzky-Golay filter length: " << savgol_window_length << ", polynomial order: " << savgol_polyorder << ", m: " << m;
    gram_sg::SavitzkyGolayFilterConfig sg_conf(m, 0, savgol_polyorder, 0);
    savgol_ = gram_sg::SavitzkyGolayFilter(sg_conf);


    f_min_bin = std::floor(f_min_() / freq_resolution);
    f_max_bin = std::ceil(f_max_() / freq_resolution);
    LOG(INFO) << name() << " IAF search range: " << f_min_() << " - " << f_max_() << " Hz (bins " << f_min_bin << " - " << f_max_bin << ")";

    max_analyze_bin = 30 / freq_resolution + 1; // Analyze up to 30 Hz to avoid high-frequency noise

    // FFTW output spectrum
    signal_in = fftwf_alloc_real(n_fft_);
    freq_half = fftwf_alloc_complex(n_fft_ / 2 + 1);

    // FFT frequency bins
    freqs.resize(max_analyze_bin);
    for (size_t k = 0; k < max_analyze_bin; ++k)
    {
        freqs[k] = static_cast<double>(k) * freq_resolution;
    }

    {
        std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
        fft_plan_ = fftwf_plan_dft_r2c_1d(n_fft_, signal_in, freq_half, FFTW_WISDOM_ONLY);
        if (fft_plan_ == nullptr)
        {
            LOG(WARNING) << name() << " No wisdom available for FFT planning, using patient mode.";
            fft_plan_ = fftwf_plan_dft_r2c_1d(n_fft_, signal_in, freq_half, FFTW_PATIENT);
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
}

void IAFEstimator::Process(ProcessingContext &context)
{
    MultiChannelType<float>::Data *data_in;
    ScalarType<double>::Data *data_out;

    // Helper lambda: one Kalman update step
    auto kalman_update = [&](double measurement) {
        // Predict: uncertainty grows
        kf_P_ = kf_P_ + kf_Q_;
        // Update: compute gain, correct estimate, shrink uncertainty
        const double K = kf_P_ / (kf_P_ + kf_R_);
        kf_x_ = kf_x_ + K * (measurement - kf_x_);
        kf_P_ = (1.0 - K) * kf_P_;
    };

    // Measurement phase
    while (!context.terminated())
    {

        if (n_messages_() != -1 && packet_count_ >= n_messages_())
        {
            break;
        }

        // Try to retrieve all new data
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
            // LOG(INFO) << name() << " Calculating IAF for packet " << packet_count_ << ", last calculated packet: " << last_calc_packet_;
            // Convert circular buffer<float> to continuous array for FFTW input and zero-pad to n_fft length
            // signal_in = sample_window.linearize();
            for (int i = 0; i < window_size_; ++i)
            {
                signal_in[i] = sample_window[i]; // Get the last 'window_size_' samples from the circular buffer
            }
            for (int i = window_size_; i < n_fft_; ++i)
            {
                signal_in[i] = 0.0f;
            }

            fftwf_execute(fft_plan_);

            std::vector<double> power(max_analyze_bin);
            // Compute power spectrum
            for (size_t k = 0; k < max_analyze_bin; ++k)
            {
                power[k] = (pow(freq_half[k][0], 2) + pow(freq_half[k][1], 2)) / (n_fft_ * fs_);
            }

            std::vector<double> power_flat(max_analyze_bin);
            power_flat = remove_aperiodic(power, freqs);

            std::vector<double> power_smooth(max_analyze_bin);
            power_smooth = savgol_filter(savgol_, power_flat);

            PeakSeed seed = find_peak_seed(power_smooth, f_min_bin, f_max_bin, freq_resolution, true);
            PeakFitResult peak = fit_gaussian_peak(power_smooth, seed, freq_resolution);

            peak.valid = bic_test(power_smooth, freqs, peak, f_min_bin, f_max_bin);

            if (peak.valid)
            {
                if (std::isnan(kf_x_))
                {
                    // First valid estimate — initialize state directly (no smoothing yet)
                    kf_x_ = peak.iaf_hz;
                }
                else
                {
                    kalman_update(peak.iaf_hz);
                }
                invalid_count_ = std::max(0, invalid_count_ - 1);
                current_iaf_ = peak.iaf_hz;
                last_valid_iaf_ = peak.iaf_hz;
                current_gauss_width_ = peak.sigma_hz;
            }
            else
            {
                invalid_count_ = std::min(invalid_count_ + 1, invalid_threshold_);
                if ((!std::isnan(last_valid_iaf_)) && (!std::isnan(kf_x_)))
                {
                    // No valid peak — fall back to last known IAF as measurement
                    kalman_update(last_valid_iaf_);
                }
                current_iaf_ = std::numeric_limits<double>::quiet_NaN();
                current_gauss_width_ = std::numeric_limits<double>::quiet_NaN();
            }
            if (invalid_count_ >= (fs_ * max_invalid_sec()) / calc_interval_())
            {
                kf_x_ = std::numeric_limits<double>::quiet_NaN(); // Reset if too many invalid estimates
                kf_P_ = kalman_full_() ? kf_R_ : std::sqrt(kf_Q_ * kf_R_);
            }

            iaf_state_->set(kf_x_);
            TimePoint end_time = Clock::now();
            if (packet_count_ % int(fs_) == 0)
            {
                LOG(INFO) << name() << " Packet " << packet_count_ << ": Estimated IAF = " << current_iaf_ << "Hz (sigma: " << current_gauss_width_ << "), KF estimate: " << kf_x_ << "Hz (P=" << kf_P_ << "), took " << std::chrono::duration<double, std::micro>(end_time - start_time).count() << " us";
            }
            // double processing_time_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
            // LOG(INFO) << name() << " Processed packet "<< packet_count_ << " in " << std::fixed << std::setprecision(2) << processing_time_us << " us";
        }
        data_out->set_data(kf_x_);
        data_out->set_source_timestamp(Clock::now());
        data_out_port_->slot(0)->PublishData();

        packet_count_++;
    }


}

void IAFEstimator::Postprocess(ProcessingContext &context)
{
    printf("\n ---------------- \n IAFEstimator: Total messages processed: %d", packet_count_);
}

void IAFEstimator::Unprepare(GlobalContext &context)
{
    // Save FFTW wisdom for future runs to speed up plan creation
    int ret = fftwf_export_wisdom_to_filename(context.resolve_path("fftw_wisdom.txt", "fft_wisdom").c_str());
    if (ret == 0)
    {
        LOG(WARNING) << name() << " Failed to save FFTW wisdom.";
    }
    {
        std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
        fftwf_destroy_plan(fft_plan_);
    }
    fftwf_free(signal_in);
    fftwf_free(freq_half);

}

REGISTERPROCESSOR(IAFEstimator);
