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

    int get_max_bin(fftwf_complex *freq, size_t N)
    {
        int max_bin = -1;
        double max_mag2 = -1.0;

        for (size_t k = 0; k < N; ++k)
        {
            double mag2 = pow(freq[k][0], 2) + pow(freq[k][1], 2);

            if (mag2 > max_mag2)
            {
                max_mag2 = mag2;
                max_bin = static_cast<int>(k);
            }
        }
        return max_bin;
    }

    int get_max_bin(const std::vector<double> &power)
    {
        int max_bin = -1;
        double max_value = -1.0f;

        for (size_t k = 0; k < power.size(); ++k)
        {
            if (power[k] > max_value)
            {
                max_value = power[k];
                max_bin = static_cast<int>(k);
            }
        }

        return max_bin;
    }

    struct LinearFitResult
    {
        float slope;
        float intercept;
        float r_squared;
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
        std::vector<double> log_freqs(freqs.size()-1);
        std::vector<double> log_power(power.size()-1);

        for (size_t k = 0; k < power.size()-1; ++k)
        {
            log_freqs[k] = std::log10(freqs[k+1]);
            log_power[k] = std::log10(power[k+1]);
        }

        LinearFitResult fit = linear_regression(log_freqs, log_power);

        std::vector<double> power_flat(power.size());
        power_flat[0] = 0; // DC component is not used for IAF estimation, set to 0 to avoid being maximum 
        for (size_t k = 0; k < power.size()-1; ++k)
        {
            double aperiodic_fit = fit.slope * log_freqs[k] + fit.intercept;
            power_flat[k+1] = pow(10, log_power[k] - aperiodic_fit);
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
    add_option("calc_interval", calc_interval_, "Number of packets between IAF calculations.");

    iaf_state_ = create_broadcaster_state<float>(
        "iaf", current_iaf_, Permission::NONE,
        "Individual alpha frequency shared with downstream processors.");
}

void IAFEstimator::CreatePorts()
{
    data_in_port_ = create_input_port<MultiChannelType<float>>(
        "in",
        MultiChannelType<float>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
        PortInPolicy(SlotRange(0, MAX_NCHANNELS)));

    data_out_port_ = create_output_port<ScalarType<float>>(
        "out",
        ScalarType<float>::Parameters(1), // Placeholder, will be set in CompleteStreamInfo
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
    window_size_ = window_size_sec_() * fs_; // 2 cycles of a 10 Hz sine wave
    n_fft_ = static_cast<int>(good_size_real(window_size_));
    sample_window.set_capacity(window_size_); // Initialize with max expected window size for 5 Hz IAF
    LOG(INFO) << name() << " Sample window size set to " << window_size_ << ", FFT size: " << n_fft_ << "\n";

    // const int n_fft = good_size_real(static_cast<size_t>(n_fft_()));
    // if (n_fft < N)
    // {
    //     throw std::runtime_error("IAFEstimator: n_fft must be >= window length");
    // }

    iaf_state_->set(current_iaf_);

    // Load FFTW wisdom if available to speed up plan creation
    fftwf_import_wisdom_from_filename(context.resolve_path("fftw_wisdom.txt", "fft_wisdom").c_str());
}


void IAFEstimator::Process(ProcessingContext &context)
{
    MultiChannelType<float>::Data *data_in;
    ScalarType<float>::Data *data_out = nullptr;

    float freq_resolution = static_cast<float>(fs_) / n_fft_;
    int savgol_window_length = static_cast<int>(2.5 / freq_resolution); // 2.5 Hz window for smoothing
    if (savgol_window_length % 2 == 0)
    {      
        savgol_window_length += 1; // Ensure window length is odd
    }

    int savgol_polyorder = 5;
    if (savgol_polyorder >= savgol_window_length)
    {
        savgol_window_length = savgol_polyorder + 2;
    }
    int m = (savgol_window_length) / 2; 
    LOG(INFO) << name() << " Savitzky-Golay filter length: " << savgol_window_length << ", polynomial order: " << savgol_polyorder << ", m: " << m << "\n";
    gram_sg::SavitzkyGolayFilterConfig sg_conf(m, 0, savgol_polyorder, 0);
    gram_sg::SavitzkyGolayFilter savgol(sg_conf);

    // FFTW output spectrum
    float *signal_in = fftwf_alloc_real(n_fft_);
    fftwf_complex *freq_half = fftwf_alloc_complex(n_fft_ / 2 + 1);

    // FFT frequency bins
    std::vector<double> freqs(n_fft_ / 2 + 1);
    for (size_t k = 0; k < n_fft_ / 2 + 1; ++k)
    {
        freqs[k] = static_cast<double>(k) * fs_ / n_fft_;
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

        // Try to retrieve data
        if (!data_in_port_->slot(0)->RetrieveData(data_in))
        {
            break;
        }
        TimePoint start_time = Clock::now();

        sample_window.push_back(data_in->data_sample(0, 0));

        data_out = data_out_port_->slot(0)->ClaimData(false);
        data_out->CloneTimestamps(*data_in);

        data_in_port_->slot(0)->ReleaseData();

        if ((sample_window.size() == sample_window.capacity()) && (packet_count_ % calc_interval_() == 0))
        {

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

            std::vector<double> power(n_fft_ / 2 + 1);
            // Compute power spectrum
            for (size_t k = 0; k < n_fft_ / 2 + 1; ++k)
            {
                power[k] = pow(freq_half[k][0], 2) + pow(freq_half[k][1], 2);
            }

            std::vector<double> power_flat(n_fft_ / 2 + 1);
            power_flat = remove_aperiodic(power, freqs);

            std::vector<double> power_smooth(n_fft_ / 2 + 1);
            power_smooth = savgol_filter(savgol, power_flat);

            // Select IAF as maximum of smoothed power spectrum
            int max_bin = get_max_bin(power_smooth);
            float max_bin_iaf = current_iaf_;
            // LOG(INFO) << name() << " Estimated IAF: " << max_bin * freq_resolution << " Hz (max bin: " << max_bin << ")\n";
            if (max_bin >= 0)
            {
                max_bin_iaf = static_cast<float>(max_bin) * freq_resolution;
            }

            current_iaf_ = max_bin_iaf;
            iaf_state_->set(current_iaf_);
            // if (std::abs(max_bin_iaf - current_iaf_) > 1e-3f)
            // {
                // printf("\n Packet %d: Estimated IAF = %.2f Hz (max bin: %d)", packet_count_, current_iaf_, max_bin);
                // } else {
                //   printf("\n IAF Estimation did not change: %.2fHz", current_iaf_);
            // }
            TimePoint end_time = Clock::now();
            // double processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            // LOG(INFO) << name() << " Processed packet "<< packet_count_ << " in " << std::fixed << std::setprecision(2) << processing_time_ms << " ms\n";
        }
        data_out->set_data(current_iaf_);
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
