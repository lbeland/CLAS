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

#include "Producer.hpp"
#include "logging/log.hpp"
#include <fstream>
#include <iomanip>
#include "threadutilities.hpp"
#include <thread>
#include <chrono>
#include <cmath>
#include <sstream>
#include <cctype>
#include <array>

namespace
{

    struct SignalState
    {
        double value;
        double amplitude;
        double theta;
        double inst_freq;
    };

    double WrapPhase(double phase)
    {
        return std::remainder(phase, 2.0 * M_PI);
    }

    SignalState ComputeSignalState(double carrier_phase,
                                   double carrier_amplitude,
                                   double carrier_frequency,
                                   const std::string &modulation_type,
                                   double modulation_amplitude,
                                   double modulation_frequency,
                                   double modulation_phase)
    {

        SignalState state{};

        if (modulation_type == "amplitude")
        {
            state.amplitude = carrier_amplitude + modulation_amplitude * std::cos(modulation_phase);
            state.theta = carrier_phase;
            state.inst_freq = carrier_frequency;
            state.value = state.amplitude * std::cos(state.theta);
            return state;
        }

        if (modulation_type == "phase")
        {
            if (modulation_frequency == 0.0)
            {
                state.amplitude = carrier_amplitude;
                state.theta = carrier_phase + modulation_amplitude;
                state.inst_freq = carrier_frequency;
                state.value = state.amplitude * std::cos(state.theta);
                return state;
            }

            state.amplitude = carrier_amplitude;
            state.theta = carrier_phase + (modulation_amplitude / modulation_frequency) * std::sin(modulation_phase);
            state.inst_freq = carrier_frequency + modulation_amplitude * std::cos(modulation_phase);
            state.value = state.amplitude * std::cos(state.theta);
            return state;
        }

        state.amplitude = carrier_amplitude;
        state.theta = carrier_phase;
        state.inst_freq = carrier_frequency;
        state.value = state.amplitude * std::cos(state.theta);
        return state;
    }

} // namespace

Producer::Producer() : IProcessor(PRIORITY_HIGH)
{
    add_option("path", path_, "Path (server-side) where to save data.");
    add_option("fs", fs_, "Sample Frequency");
    add_option("carrier_amplitude", carrier_amplitude_, "Carrier signal amplitude.");
    add_option("carrier_frequency", carrier_frequency_, "Carrier signal frequency in Hz.");
    add_option("modulation_type", modulation_type_, "Modulation type: none, amplitude, or phase.");
    add_option("modulation_amplitude", modulation_amplitude_, "Modulation amplitude.");
    add_option("modulation_frequency", modulation_frequency_, "Modulation frequency in Hz.");
    add_option("nchannels", nchannels_, "Number of channels to generate.");
    add_option("nsamples", nsamples_, "Number of samples per packet.");
    add_option("n_messages", n_messages_, "Number of packets to generate (-1 = infinite).");

    iaf_state_ = create_broadcaster_state<double>(
        "iaf", current_iaf_, Permission::NONE,
        "Individual alpha frequency shared with downstream processors.");
}

void Producer::CreatePorts()
{
    data_out_port_ = create_output_port<MultiChannelType<float>>(
        "out",
        MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()),
        PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));
    
    meta_out_port_ = create_output_port<MultiChannelType<double>>(
        "meta_out",
        MultiChannelType<double>::Parameters(3,1, fs_()),
        PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));
}

void Producer::CompleteStreamInfo()
{
    data_out_port_->slot(0)->streaminfo().set_parameters(
        MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()));
    data_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());

    meta_out_port_->slot(0)->streaminfo().set_parameters(
        MultiChannelType<double>::Parameters(3,1, fs_()));
    meta_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());
}

void Producer::Preprocess(ProcessingContext &context)
{
    // send_times.clear();
    packet_count_ = 0;
}

void Producer::Process(ProcessingContext &context)
{
    MultiChannelType<float>::Data *data_out = nullptr;
    MultiChannelType<double>::Data *meta_out = nullptr;
    
    double carrier_phase = 0.0;
    double modulation_phase = 0.0;
    TimePoint timestamp;
    uint64_t hardware_time_us = 0;

    const double carrier_step = 2.0 * M_PI * carrier_frequency_() / fs_();
    const double modulation_step = 2.0 * M_PI * modulation_frequency_() / fs_();

    // Use wall clock time as reference for hardware timestamps
    uint64_t start_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    while (!context.terminated())
    {
        if (n_messages_() != -1 && packet_count_ >= n_messages_())
        {
            break;
        }

        data_out = data_out_port_->slot(0)->ClaimData(false);
        meta_out = meta_out_port_->slot(0)->ClaimData(false);

        const SignalState state = ComputeSignalState(
            carrier_phase,
            carrier_amplitude_(),
            carrier_frequency_(),
            modulation_type_(),
            modulation_amplitude_(),
            modulation_frequency_(),
            modulation_phase);

        std::vector<double> meta_data = {
            state.amplitude,
            state.theta,
            state.inst_freq,
        };

        current_iaf_ = static_cast<float>(state.inst_freq);
        iaf_state_->set(current_iaf_);


        // Add packet count to start time for source timestamp
        timestamp = Clock::now();
        hardware_time_us = start_time + (uint64_t)packet_count_ * 1000000ULL / fs_();

        std::vector<float> sample_vec(nchannels_(), static_cast<float>(state.value));
        data_out->set_data_sample(0, sample_vec);
        data_out->set_sample_timestamp(0, hardware_time_us);
        data_out->set_source_timestamp(timestamp);
        data_out->set_hardware_timestamp(hardware_time_us);
        data_out_port_->slot(0)->PublishData();

        meta_out->set_data_sample(0, meta_data);
        meta_out->set_sample_timestamp(0, hardware_time_us);
        meta_out->set_source_timestamp(timestamp);
        meta_out->set_hardware_timestamp(hardware_time_us);
        meta_out_port_->slot(0)->PublishData();

        // send_times.push_back(timestamp);
        ++packet_count_;
        // custom_sleep_for(90);

        carrier_phase = WrapPhase(carrier_phase + carrier_step);
        modulation_phase = WrapPhase(modulation_phase + modulation_step);
    }
    LOG(INFO) << name() << " stopped working";

}

void Producer::Postprocess(ProcessingContext &context)
{
    std::ostringstream statistic_print;

    statistic_print << "\n ---------------- \n Total messages sent: " << packet_count_;

    // if (send_times.empty())
    // {
    //     return;
    // }

    // double sum_diff_us = 0.0;
    // double max_diff_us = 0.0;
    // std::size_t max_idx = 0;
    // double sum_sq_diff = 0.0;
    // std::vector<double> send_times_diff;
    // const int start_idx = 0;
    // const int n_times = static_cast<int>(send_times.size()) - start_idx;
    // if (n_times <= 0)
    // {
    //     statistic_print << "\n Not enough messages to calculate statistics.";
    //     std::cout << statistic_print.str();
    //     return;
    // }

    // send_times_diff.resize(static_cast<std::size_t>(n_times - 1));

    // for (std::size_t i = static_cast<std::size_t>(start_idx); i + 1 < send_times.size(); ++i)
    // {
    //     const double diff_us = std::chrono::duration<double, std::micro>(send_times[i + 1] - send_times[i]).count();
    //     send_times_diff[i - static_cast<std::size_t>(start_idx)] = diff_us;
    //     sum_diff_us += diff_us;
    //     sum_sq_diff += diff_us * diff_us;
    //     if (diff_us > max_diff_us)
    //     {
    //         max_diff_us = diff_us;
    //         max_idx = i;
    //     }
    // }

    // const std::size_t n_periods = send_times_diff.size();
    // double avg_period = 0.0;
    // double std_period = 0.0;
    // if (n_periods > 0)
    // {
    //     avg_period = sum_diff_us / static_cast<double>(n_periods);
    //     const double variance = (sum_sq_diff / static_cast<double>(n_periods)) - (avg_period * avg_period);
    //     std_period = std::sqrt(std::max(0.0, variance));
    // }

    // statistic_print << "\n Average send period (us): " << avg_period;
    // statistic_print << "\n Max send period (us): " << max_diff_us << ", idx: " << max_idx;
    // statistic_print << "\n Std send period (us): " << std_period << "\n";

    // const std::string append = "Producer.csv";
    // std::ofstream output;
    // std::string filename = context.resolve_path(path_(), "run");
    // output.open(filename + append);
    // output << "Metric,Send_period\n";
    // output << "mean," << avg_period << "\n";
    // output << "std," << std_period << "\n";
    // output << "max," << max_diff_us << "\n";
    // output.close();

    // const std::string send_times_append = "send_times.csv";
    // std::ofstream send_times_output;
    // send_times_output << std::fixed << std::setprecision(17);
    // send_times_output.open(filename + send_times_append);
    // for (double t : send_times_diff)
    // {
    //     send_times_output << t << "\n";
    // }
    // send_times_output.close();

    // std::cout << statistic_print.str();
}

REGISTERPROCESSOR(Producer);
