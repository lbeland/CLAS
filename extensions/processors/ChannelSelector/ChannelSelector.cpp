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

#include "ChannelSelector.hpp"
#include "utilities/time.hpp"
#include "logging/log.hpp"
#include <chrono>
#include <algorithm>
#include <limits>
#include <cmath>
#include <string>

ChannelSelector::ChannelSelector() : IProcessor(PRIORITY_HIGH)
{
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("channel_indices", channel_indices_, "Comma-separated list of channel indices to select from (1-based).");
    add_option("rms_window_seconds", rms_window_seconds_, "Length of the weighted RMS window in seconds (recent samples get higher weights).");

    channel_state_ = create_broadcaster_state<unsigned int>(
        "ch_idx", current_channel_index_, Permission::NONE,
        "Current selected channel index (zero-based) shared with downstream processors.");
}

void ChannelSelector::CreatePorts()
{
    data_in_port_ = create_input_port<MultiChannelType<float>>(
        "in",
        MultiChannelType<float>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
        PortInPolicy(SlotRange(0, MAX_NCHANNELS)));

    // data_out_port_ = create_output_port<MultiChannelType<float>>(
    //     "out",
    //     MultiChannelType<float>::Parameters(1, 1, 1), // Placeholder, will be set in CompleteStreamInfo
    //     PortOutPolicy(SlotRange(0, MAX_NCHANNELS), 200, WaitStrategy::kBlockingStrategy));

    idx_out_port = create_output_port<ScalarType<unsigned int>>(
        "ch_idx_out",
        ScalarType<unsigned int>::Parameters(1), // Placeholder, will be set in CompleteStreamInfo
        PortOutPolicy(SlotRange(0, MAX_NCHANNELS), 200, WaitStrategy::kBlockingStrategy));
}

void ChannelSelector::CompleteStreamInfo()
{
    const auto &input_params = data_in_port_->slot(0)->streaminfo().parameters<MultiChannelType<float>::Parameters>();

    // // only pass through the selected channel, so set output nchannels to 1 but keep nsamples and sample_rate the same as input
    // data_out_port_->streaminfo(0).set_parameters(MultiChannelType<float>::Parameters(1, input_params.nsamples, input_params.sample_rate));
    // data_out_port_->streaminfo(0).set_stream_rate(data_in_port_->streaminfo(0));

    // Set the parameters for the channel index output port
    idx_out_port->streaminfo(0).set_parameters(ScalarType<unsigned int>::Parameters(1));
    idx_out_port->streaminfo(0).set_stream_rate(data_in_port_->streaminfo(0));
}

void ChannelSelector::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);
    const auto &p = info.parameters<MultiChannelType<float>::Parameters>();
    LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate << "\n";

    fs_ = p.sample_rate;
    n_channels_ = p.nchannels;

    selected_channels_.clear();
    const auto &configured_channels = channel_indices_();
    if (configured_channels.empty())
    {
        selected_channels_.reserve(n_channels_);
        for (unsigned int channel_idx = 0; channel_idx < n_channels_; ++channel_idx)
        {
            selected_channels_.push_back(channel_idx);
        }
    }
    else
    {
        selected_channels_.reserve(configured_channels.size());
        for (int channel_idx : configured_channels)
        {
            if (channel_idx <= 0 || static_cast<unsigned int>(channel_idx) > n_channels_)
            {
                throw std::runtime_error(name() + ": channel_indices contains out-of-range index " + std::to_string(channel_idx));
            }
            selected_channels_.push_back(static_cast<unsigned int>(channel_idx) - 1); // convert from 1-based to 0-based index
        }
    }

    const double tau_seconds = rms_window_seconds_();
    // EMA
    ema_mu_ = std::exp(-1.0 / (fs_ * tau_seconds));
}

void ChannelSelector::Preprocess(ProcessingContext &context)
{
    current_channel_index_ = selected_channels_.front();
    channel_state_->set(current_channel_index_);

    // Assign initial value as the exact threshold squared
    ema_.assign(n_channels_, rms_threshold_uv_() * rms_threshold_uv_());

    LOG(INFO) << name() << " RMS selector EMA tau: " << rms_window_seconds_()
              << " s, mu: " << ema_mu_ << ".";

    packet_count_ = 0;
}

void ChannelSelector::Process(ProcessingContext &context)
{
    MultiChannelType<float>::Data *data_in = nullptr;
    // MultiChannelType<float>::Data *data_out = nullptr;
    ScalarType<unsigned int>::Data *idx_out = nullptr;

    double rms_thresh_uv = rms_threshold_uv_();

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

        double best_mean_square = -std::numeric_limits<double>::infinity();

        // First check of current channel is still above threshold, if not, loop through all channels to find the one with highest RMS
        const double sample = static_cast<double>(data_in->data_sample(0, current_channel_index_));
        const double sample_square = sample * sample;
        const double ema_current = ema_mu_ * ema_[current_channel_index_] + (1 - ema_mu_) * sample_square;

        // Because true RMS would take the sqrt, we compare to squared threshold
        if (ema_current > rms_thresh_uv * rms_thresh_uv)
        {
            ema_[current_channel_index_] = ema_current;
            best_mean_square = ema_current;
        }
        else
        {
            // LOG(DEBUG) << name() << packet_count_ << " Channel " << current_channel_index_ + 1 << " RMS " << std::sqrt(ema_current) << " uV below threshold " << rms_thresh_uv << " uV, checking other channels...";
            for (unsigned int channel_idx : selected_channels_)
            {
                const double sample = static_cast<double>(data_in->data_sample(0, channel_idx));
                const double sample_square = sample * sample;
                ema_[channel_idx] = ema_mu_ * ema_[channel_idx] + (1 - ema_mu_) * sample_square;
                if (ema_[channel_idx] > best_mean_square)
                {
                    best_mean_square = ema_[channel_idx];
                    current_channel_index_ = channel_idx;
                }
            }
        }

        channel_state_->set(current_channel_index_);

        if (packet_count_ % int(5 * fs_) == 0)
        {
            LOG(INFO) << name() << ". Packet " << packet_count_ + 1 << ": Selected channel " << current_channel_index_ + 1 << " (RMS: " << std::sqrt(ema_[current_channel_index_]) << "uV)";
        }

        // Claim output buffer
        // data_out = data_out_port_->slot(0)->ClaimData(false);
        idx_out = idx_out_port->slot(0)->ClaimData(false);

        // data_out->set_data_sample(0, 0, data_in->data_sample(0, current_channel_index_));
        // data_out->set_sample_timestamps(data_in->sample_timestamps());
        // data_out->CloneTimestamps(*data_in);

        idx_out->set_data(current_channel_index_ + 1); // convert back to 1-based index
        idx_out->set_hardware_timestamp(data_in->hardware_timestamp());
        idx_out->set_source_timestamp(Clock::now());

        data_in_port_->slot(0)->ReleaseData();
        // data_out_port_->slot(0)->PublishData();
        idx_out_port->slot(0)->PublishData();

        packet_count_++;
    }
    LOG(INFO) << name() << " stopped working";

}

void ChannelSelector::Postprocess(ProcessingContext &context)
{
    printf("\n ---------------- \n ChannelSelector: Total messages processed: %d, last selected channel: %u",
           packet_count_, current_channel_index_);
}

REGISTERPROCESSOR(ChannelSelector);
