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

    int get_max_bin(const std::vector<float> &power)
    {
        int max_bin = -1;
        float max_value = -1.0f;

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
    add_option("num_segments", num_segments_, "Number of segments for Welch's method.");
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
    int N = static_cast<int>(fs_ * window_size_sec_()); // 5 seconds
    sample_window.set_capacity(N);
    LOG(INFO) << name() << " Sample window size set to " << sample_window.capacity() << "\n";

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

    // FFTW output spectrum
    const int N = static_cast<int>(sample_window.capacity());

    const int kNumSegments = num_segments_();    // Use 10 segments for Welch's method by default

    const int nperseg = static_cast<int>(N / kNumSegments); 

    const int overlap = static_cast<int>(nperseg / 2); // Desired overlap as half of segment length
    const int hop_length = nperseg - overlap;
    LOG(INFO) << name() << " Segment overlap: " << overlap << ", N per segment: " << nperseg << ", Hop length: " << hop_length << "\n";

    std::vector<float> hann_window(nperseg, 1.0f);
    const double pi = std::acos(-1.0);
    const double denom = static_cast<double>(nperseg - 1);
    for (int i = 0; i < nperseg; ++i)
    {
        hann_window[i] = static_cast<float>(0.5 * (1.0 - std::cos((2.0 * pi * i) / denom)));
    }

    const int n_fft = good_size_real(static_cast<size_t>(nperseg));
    LOG(INFO) << name() << " Using n_fft = " << n_fft;
    float *signal_batch = fftwf_alloc_real(kNumSegments * n_fft);
    fftwf_complex *freq_batch = fftwf_alloc_complex(kNumSegments * (n_fft / 2 + 1));

    int fft_size[1] = {n_fft};
    fftwf_plan p;
    {
        std::lock_guard<std::mutex> lock(dsp::fftw::planner_mutex);
        p = fftwf_plan_many_dft_r2c(
            1,
            fft_size,
            kNumSegments,
            signal_batch,
            nullptr,
            1,
            n_fft,
            freq_batch,
            nullptr,
            1,
            (n_fft / 2) + 1,
            FFTW_WISDOM_ONLY);
        if (p == nullptr)
        {
            LOG(WARNING) << name() << "No wisdom available for FFT planning, using patient mode.";
            p = fftwf_plan_many_dft_r2c(
                1,
                fft_size,
                kNumSegments,
                signal_batch,
                nullptr,
                1,
                n_fft,
                freq_batch,
                nullptr,
                1,
                (n_fft / 2) + 1,
                FFTW_PATIENT);
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
            std::vector<float> averaged_power(n_fft / 2 + 1, 0.0f);

            for (int segment = 0; segment < kNumSegments; ++segment)
            {
                int start = segment * hop_length;
                if (start + nperseg > N)
                {
                    start = std::max(0, N - nperseg);
                }

                float *segment_input = signal_batch + (segment * n_fft);
                std::fill(segment_input, segment_input + n_fft, 0.0f);
                for (int i = 0; i < nperseg && (start + i) < N; ++i)
                {
                    segment_input[i] = sample_window[start + i] * hann_window[i];
                }
            }

            fftwf_execute(p);

            for (int segment = 0; segment < kNumSegments; ++segment)
            {
                // Quadratic weighting to give more emphasis to later segments in the window, which are more recent
                const fftwf_complex *segment_freq = freq_batch + (segment * (n_fft / 2 + 1));
                for (int bin = 0; bin <= n_fft / 2; ++bin)
                {
                    const float re = segment_freq[bin][0];
                    const float im = segment_freq[bin][1];
                    averaged_power[bin] += re * re + im * im;
                }
            }

            // Select IAF directly from the maximum bin of Welch power spectrum.
            int max_bin = get_max_bin(averaged_power);
            float freq_resolution = static_cast<float>(fs_) / n_fft;
            const float max_bin_iaf = (max_bin >= 0) ? (static_cast<float>(max_bin) * freq_resolution) : current_iaf_;
            if (std::abs(max_bin_iaf - current_iaf_) > 1e-3f)
            {
                current_iaf_ = max_bin_iaf;
                // printf("\n Packet %d: Estimated IAF = %.2f Hz (max bin: %d)", packet_count_, current_iaf_, max_bin);

                iaf_state_->set(current_iaf_);

                // } else {
                //   printf("\n IAF Estimation did not change: %.2fHz", current_iaf_);
            }
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
    fftwf_free(signal_batch);
    fftwf_free(freq_batch);
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
