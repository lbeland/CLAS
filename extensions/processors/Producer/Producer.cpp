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
#include "threadutilities.hpp"
#include <thread>
#include <chrono>
#include <cmath>
#include <sstream>
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

        // Default: no modulation
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
    add_option("fs", fs_, "Sample frequency (Hz).");
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
    data_out_port_ = create_output_port<MultiChannelType<double>>(
        "out",
        MultiChannelType<double>::Parameters(nchannels_(), nsamples_(), fs_()),
        PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));

    meta_out_port_ = create_output_port<MultiChannelType<double>>(
        "meta_out",
        MultiChannelType<double>::Parameters(3, 1, fs_()),
        PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));
}

void Producer::CompleteStreamInfo()
{
    data_out_port_->slot(0)->streaminfo().set_parameters(
        MultiChannelType<double>::Parameters(nchannels_(), nsamples_(), fs_()));
    data_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());

    meta_out_port_->slot(0)->streaminfo().set_parameters(
        MultiChannelType<double>::Parameters(3, 1, fs_()));
    meta_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());
}

void Producer::Preprocess(ProcessingContext &context)
{
    packet_count_ = 0;
}

void Producer::Process(ProcessingContext &context)
{
    MultiChannelType<double>::Data *data_out = nullptr;
    MultiChannelType<double>::Data *meta_out = nullptr;

    std::vector<double> sample_vec(nchannels_());

    double carrier_phase = 0.0;
    double modulation_phase = 0.0;
    uint64_t hardware_time_us = 0;

    const double carrier_step = 2.0 * M_PI * carrier_frequency_() / fs_();
    const double modulation_step = 2.0 * M_PI * modulation_frequency_() / fs_();

    // Base time in steady_clock (same domain as source_timestamp / Clock::now()).
    // The wallclock offset converts it to UTC µs for hardware_timestamp, matching SourceClient.
    uint64_t start_time = std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now().time_since_epoch()).count();
    TimePoint start_time_point = Clock::now();
    int64_t steady_to_wallclock_offset_us =
        static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) -
        static_cast<int64_t>(start_time);

    // Expected wall-clock gap between consecutive packets, used only to pace emission.
    const double inter_packet_ns = (static_cast<double>(nsamples_()) / fs_()) * 1e9;

    // last_emit_time_ = Clock::now();

    while (!context.terminated())
    {
        if (n_messages_() != -1 && packet_count_ >= n_messages_())
        {
            break;
        }

        // const TimePoint target_emit_time =
        //     last_emit_time_ + std::chrono::nanoseconds(static_cast<int64_t>(inter_packet_ns));
        const TimePoint target_emit_time = start_time_point + packet_count_ * std::chrono::nanoseconds(static_cast<int64_t>(inter_packet_ns));

        const TimePoint now = Clock::now();
        if (target_emit_time > now)
        {
            constexpr int64_t puffer = 5;
            const int64_t remaining_us = std::chrono::duration_cast<std::chrono::microseconds>(
                target_emit_time - now).count();
            const int64_t sleep_for = std::max<int64_t>(remaining_us - puffer, 0);
            custom_sleep_for(static_cast<uint64_t>(sleep_for));
        }

        SignalState state = ComputeSignalState(
            carrier_phase,
            carrier_amplitude_(),
            carrier_frequency_(),
            modulation_type_(),
            modulation_amplitude_(),
            modulation_frequency_(),
            modulation_phase);

        // Add random noise (10% of amplitude)
        state.value += 0.1 * carrier_amplitude_() * ((std::rand() / (double)RAND_MAX) - 0.5);

        std::vector<double> meta_data = {state.amplitude, state.theta, state.inst_freq};

        current_iaf_ = state.inst_freq;
        iaf_state_->set(current_iaf_);

        // hardware_time_us is in steady_clock µs; convert to wall-clock for hardware_timestamp
        hardware_time_us = start_time + (uint64_t)packet_count_ * 1000000ULL / fs_();

        std::fill(sample_vec.begin(), sample_vec.end(), state.value);

        data_out = data_out_port_->slot(0)->ClaimData(false);

        data_out->set_data_sample(0, sample_vec);
        data_out->set_sample_timestamp(0, hardware_time_us + steady_to_wallclock_offset_us);
        data_out->set_source_timestamp(TimePoint(std::chrono::microseconds(hardware_time_us)));
        data_out->set_hardware_timestamp(hardware_time_us + steady_to_wallclock_offset_us);
        data_out_port_->slot(0)->PublishData();

        meta_out = meta_out_port_->slot(0)->ClaimData(false);
        meta_out->set_data_sample(0, meta_data);
        meta_out->set_sample_timestamp(0, hardware_time_us + steady_to_wallclock_offset_us);
        meta_out->set_source_timestamp(TimePoint(std::chrono::microseconds(hardware_time_us)));
        meta_out->set_hardware_timestamp(hardware_time_us + steady_to_wallclock_offset_us);
        meta_out_port_->slot(0)->PublishData();

        ++packet_count_;

        // Re-anchor from the actual emit time (not the fixed schedule), so the next
        // iteration's pacing reflects reality rather than trying to catch up.
        // last_emit_time_ = Clock::now();

        carrier_phase = WrapPhase(carrier_phase + carrier_step);
        modulation_phase = WrapPhase(modulation_phase + modulation_step);
    }
    LOG(INFO) << name() << " stopped working";
}

void Producer::Postprocess(ProcessingContext &context)
{
    LOG(INFO) << name() << ": Total messages sent: " << packet_count_;
}

REGISTERPROCESSOR(Producer);
