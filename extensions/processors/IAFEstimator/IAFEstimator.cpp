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
#include <gram_savitzky_golay/gram_savitzky_golay.h>
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
            gauss[k] = amplitude * std::exp(-0.5 * pow((freqs[k] - center) / width, 2));
        }
        return gauss;
    }

    std::pair<double, double> fit_peak(const std::vector<double> &power, const std::vector<double> &freqs, int f_min_bin, int f_max_bin, double resolution, bool parabolic = true)
    {
        int max_bin = -1;
        double max_value = -1.0;
        double delta = 0.0;
        double iaf = -1.0;
        size_t n = f_max_bin - f_min_bin + 1;

        for (size_t k = f_min_bin; k <= f_max_bin; ++k)
        {
            // Decide between greater or greater/equal based on whether we want first peak or last peak in case of ties
            if (power[k] > max_value)
            {
                max_value = power[k];
                max_bin = static_cast<int>(k);
                iaf = max_bin * resolution;
            }
        }
        
        if (parabolic && max_bin > f_min_bin && max_bin < f_max_bin)
        {
            double y1 = power[max_bin - 1];
            double y2 = power[max_bin];
            double y3 = power[max_bin + 1];
            double denom = (y1 - 2 * y2 + y3);
            if (denom != 0){
                delta = 0.5 * (y1 - y3) / denom;
                iaf += delta * resolution;
            }
        }

        double half_max = max_value / 2.0;
        int left_half_bin = max_bin;
        while (left_half_bin > f_min_bin && power[left_half_bin] > half_max)
        {            --left_half_bin;
        }
        if (left_half_bin == f_min_bin)
        {
            // LOG(WARNING) << "Left half max not found for IAF bin: " << iaf << " Hz, max value: " << max_value;
            return std::make_pair(-1.0, -1.0); // No significant peak
        }
        int right_half_bin = max_bin;
        while (right_half_bin < f_max_bin && power[right_half_bin] > half_max)
        {            ++right_half_bin;
        }
        if (right_half_bin == f_max_bin)
        {
            // LOG(WARNING) << "Right half max not found for IAF bin: " << iaf << " Hz, max value: " << max_value;
            return std::make_pair(-1.0, -1.0); // No significant peak
        }
        double fwhm = (right_half_bin - left_half_bin) * resolution;
        double std_gauss = fwhm / (2 * std::sqrt(2 * std::log(2)));

        // Null hypothesis: flat spectrum (power=1 after aperiodic removal)
        double ss_h0 = 0.0;
        for (size_t k = f_min_bin; k <= f_max_bin; ++k)
        {
            ss_h0 += pow(power[k] - 1.0, 2);
        }
        // Catch if the spectrum is completely flat (no variance)
        if (ss_h0/n - pow(power[0] - 1.0, 2) == 0.0)
        {
            // LOG(WARNING) << "No valid IAF bin found. Max value: " << max_value << ", BIC H0: " << bic_h0;
            return std::make_pair(-1.0, -1.0); // No significant peak
        }
    
        // Alternative hypothesis: Gaussian peak on top of flat spectrum
        std::vector<double> gauss = gaussian(freqs, max_value - 1.0, iaf, std_gauss);

        // for (int k = max_bin - 5; k <= max_bin + 5; ++k){
        //     LOG(INFO) << "power_smooth[" << k << "] = " << power[k] << ", gauss[" << k << "] = " << gauss[k];
        // }


        double ss_h1 = 0.0;
        for (size_t k = f_min_bin; k <= f_max_bin; ++k)
        {
            ss_h1 += pow(power[k] - 1.0 - gauss[k], 2);
        }
        double bic_h0 = n * std::log(ss_h0/n);
        double bic_h1 = n * std::log(ss_h1/n) + 3 * std::log(n);

        if (bic_h0 - bic_h1 < 0)
        {
            // LOG(WARNING) << "No valid IAF bin found. Gauss parameters:" << iaf << ", " <<std_gauss << ", " << right_half_bin << "," << left_half_bin << ", " << max_value << ", BIC H0: " << bic_h0 << ", BIC H1: " << bic_h1;
            iaf = -1.0; // No significant peak
        }
        else {
            // LOG(INFO) << "IAF bin found: " << iaf << " Hz, std: " << std_gauss << " Hz, BIC H0: " << bic_h0 << ", BIC H1: " << bic_h1;
        }

        return std::make_pair(iaf, std_gauss);
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
        for (int i = 0; i < n; i++) {
            ss_xy += (x[i] - x_mean) * (y[i] - y_mean);
            ss_xx += (x[i] - x_mean) * (x[i] - x_mean);
            ss_yy += (y[i] - y_mean) * (y[i] - y_mean);
        }

        double slope     = ss_xy / ss_xx;
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
            safe_power[k] = std::max(power[k],1e-12);
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

        // LOG(INFO) << "Aperiodic fit: slope = " << fit.slope << ", intercept = " << fit.intercept << ", R^2 = " << fit.r_squared << "\n";

        std::vector<double> power_flat(N);
        power_flat[0] = 1.0;
        for (size_t k = 1; k < N; ++k)
        {
            double aperiodic_fit = fit.slope * log_freqs[k - 1] + fit.intercept;
            power_flat[k] = safe_power[k] / std::pow(10, aperiodic_fit);
        }
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
        for (int i=0; i < power.size(); i++)
        {
            for (int j = i; j < i+window_size; ++j) {
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
    add_option("ema_window_seconds", ema_window_seconds_, "Window size in seconds for RMS calculation used in IAF estimation.");

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
        data_out_port_->streaminfo(k).set_stream_rate(data_in_port_->streaminfo(0).stream_rate());
    }
}

void IAFEstimator::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);
    const auto &p = info.parameters<MultiChannelType<float>::Parameters>();
    LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate << "\n";
    fs_ = p.sample_rate;
    window_size_ = window_size_sec_() * fs_; // Convert window size from seconds to samples
    n_fft_ = static_cast<int>(good_size_real(window_size_));
    sample_window.set_capacity(n_fft_); // Initialize window size with next fast fft len 
    LOG(INFO) << name() << " Sample window size set to " << window_size_ << ", FFT size: " << n_fft_ << "\n";

    iaf_state_->set(current_iaf_);

    const double tau_seconds = ema_window_seconds_();
    ema_alpha_ = 1.0 - std::exp(-1.0 / (p.sample_rate * tau_seconds));
    ema_= 10.0; // Initialize for 10Hz IAF
  
    LOG(INFO) << name() << " RMS selector EMA tau: " << ema_window_seconds_()
              << " s, alpha: " << ema_alpha_ << ".";

    // Load FFTW wisdom if available to speed up plan creation
    fftwf_import_wisdom_from_filename(context.resolve_path("fftw_wisdom.txt", "fft_wisdom").c_str());
}


void IAFEstimator::Process(ProcessingContext &context)
{
    MultiChannelType<float>::Data *data_in;
    ScalarType<double>::Data *data_out;

    double freq_resolution = fs_ / n_fft_;
    LOG(INFO) << name() << " Frequency resolution: " << freq_resolution << " Hz\n";
    int savgol_window_length = static_cast<int>(2.5 / freq_resolution); // 2.5 Hz window for smoothing
    if (savgol_window_length % 2 == 0)
    {      
        savgol_window_length += 1; // Ensure window length is odd
    }

    int savgol_polyorder = 5;
    if (savgol_polyorder >= savgol_window_length)
    {
        savgol_window_length = savgol_polyorder + 2;    // Ensure window length is greater than polynomial order
    }
    int m = (savgol_window_length) / 2; 
    LOG(INFO) << name() << " Savitzky-Golay filter length: " << savgol_window_length << ", polynomial order: " << savgol_polyorder << ", m: " << m << "\n";
    gram_sg::SavitzkyGolayFilterConfig sg_conf(m, 0, savgol_polyorder, 0);
    gram_sg::SavitzkyGolayFilter savgol(sg_conf);

    int f_min_bin = std::floor(f_min_() / freq_resolution);
    int f_max_bin = std::ceil(f_max_() / freq_resolution);
    LOG(INFO) << name() << " IAF search range: " << f_min_() << " - " << f_max_() << " Hz (bins " << f_min_bin << " - " << f_max_bin << ")\n";

    int max_analyze_bin = 30 / freq_resolution + 1; // Analyze up to 30 Hz to avoid high-frequency noise

    // FFTW output spectrum
    float *signal_in = fftwf_alloc_real(n_fft_);
    fftwf_complex *freq_half = fftwf_alloc_complex(n_fft_ / 2 + 1);

    // FFT frequency bins
    std::vector<double> freqs(max_analyze_bin);
    for (size_t k = 0; k < max_analyze_bin; ++k)
    {
        freqs[k] = static_cast<double>(k) * freq_resolution;
    }

    fftwf_plan p;
    
    {
        std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
        p = fftwf_plan_dft_r2c_1d(n_fft_, signal_in, freq_half,  FFTW_WISDOM_ONLY);
        if (p == nullptr)
        {
            LOG(WARNING) << name() << "No wisdom available for FFT planning, using patient mode.";
            p = fftwf_plan_dft_r2c_1d(n_fft_, signal_in, freq_half,  FFTW_PATIENT);
        }
    }

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
            // LOG(INFO) << name() << " Calculating IAF for packet " << packet_count_ << ", last calculated packet: " << last_calc_packet_ << "\n";
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

            fftwf_execute(p);

            std::vector<double> power(max_analyze_bin);
            // Compute power spectrum
            for (size_t k = 0; k < max_analyze_bin; ++k)
            {
                power[k] = (pow(freq_half[k][0], 2) + pow(freq_half[k][1], 2)) / (n_fft_*fs_);
            }

            std::vector<double> power_flat(max_analyze_bin);
            power_flat = remove_aperiodic(power, freqs);

            std::vector<double> power_smooth(max_analyze_bin);
            power_smooth = savgol_filter(savgol, power_flat);

            // Select IAF as maximum of smoothed power spectrum
            std::pair<double, double> res = fit_peak(power_smooth, freqs, f_min_bin, f_max_bin, freq_resolution, true);
            double iaf = res.first;
            double std_gauss = res.second;
            if (iaf == -1.0)
            {
                LOG(WARNING) << name() << "Packet count " << packet_count_ << ": No valid IAF bin found between " << f_min_() << " and " << f_max_() << " Hz.";
                current_iaf_ = std::numeric_limits<double>::quiet_NaN();
            }
            else {
                current_iaf_ = iaf;
            }


            iaf_state_->set(current_iaf_);
            // if (std::abs(iaf - current_iaf_) > 1e-3f)
            // {
            //     printf("\n Packet %d: Estimated IAF = %.2f Hz (max bin: %d)", packet_count_, current_iaf_, res.first);
            //     } else {
            //       printf("\n IAF Estimation did not change: %.2fHz", current_iaf_);
            // }
            TimePoint end_time = Clock::now();
            if (packet_count_ % int(fs_) == 0)
            {
                LOG(INFO) << name() << " Packet " << packet_count_ << ": Estimated IAF = " << current_iaf_ << " Hz (std: " << std_gauss << "), took " << std::chrono::duration<double, std::micro>(end_time - start_time).count() << " us";
            }
            // double processing_time_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
            // LOG(INFO) << name() << " Processed packet "<< packet_count_ << " in " << std::fixed << std::setprecision(2) << processing_time_us << " us\n";
            
        }
        data_out->set_data(current_iaf_);
        data_out->set_source_timestamp(Clock::now());
        data_out_port_->slot(0)->PublishData();

        packet_count_++;
    }

    {
        std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
        fftwf_destroy_plan(p);
    }
    fftwf_free(signal_in);
    fftwf_free(freq_half);
}

void IAFEstimator::Postprocess(ProcessingContext &context)
{
    printf("\n ---------------- \n IAFEstimator: Total messages processed: %d", packet_count_);
}

void IAFEstimator::Unprepare(GlobalContext &context)
{
    // Save FFTW wisdom for future runs to speed up plan creation
    fftwf_export_wisdom_to_filename(context.resolve_path("fftw_wisdom.txt", "fft_wisdom").c_str());
}

REGISTERPROCESSOR(IAFEstimator);
