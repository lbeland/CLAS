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
    add_option("rms_threshold_uv", rms_threshold_uv_, "RMS threshold in microvolts for switching channels (lower = more sensitive).");

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

    idx_out_port_ = create_output_port<ScalarType<unsigned int>>(
        "ch_idx_out",
        ScalarType<unsigned int>::Parameters(1),
        PortOutPolicy(SlotRange(0, MAX_NCHANNELS), 200, WaitStrategy::kBlockingStrategy));
}

void ChannelSelector::CompleteStreamInfo()
{
    idx_out_port_->streaminfo(0).set_parameters(ScalarType<unsigned int>::Parameters(1));
    idx_out_port_->streaminfo(0).set_stream_rate(data_in_port_->streaminfo(0));
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
            selected_channels_.push_back(static_cast<unsigned int>(channel_idx) - 1); // convert from 1-based to 0-based
        }
    }

    const double tau_seconds = rms_window_seconds_();
    ema_mu_ = std::exp(-1.0 / (fs_ * tau_seconds));
}

void ChannelSelector::Preprocess(ProcessingContext &context)
{
    current_channel_index_ = selected_channels_.front();
    channel_state_->set(current_channel_index_);

    // Initialize EMA at the squared threshold so all channels start neutral
    ema_.assign(n_channels_, rms_threshold_uv_() * rms_threshold_uv_());

    LOG(INFO) << name() << " RMS selector EMA tau: " << rms_window_seconds_()
              << " s, mu: " << ema_mu_ << ".";

    packet_count_ = 0;
}

void ChannelSelector::Process(ProcessingContext &context)
{
    MultiChannelType<float>::Data *data_in = nullptr;
    ScalarType<unsigned int>::Data *idx_out = nullptr;

    const double rms_thresh_uv = rms_threshold_uv_();

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

        // Update EMA for all selected channels so no channel goes stale
        for (unsigned int channel_idx : selected_channels_)
        {
            const double s = static_cast<double>(data_in->data_sample(0, channel_idx));
            ema_[channel_idx] = ema_mu_ * ema_[channel_idx] + (1.0 - ema_mu_) * s * s;
        }

        // Compare squared EMA to squared threshold (avoids sqrt every packet)
        if (ema_[current_channel_index_] <= rms_thresh_uv * rms_thresh_uv)
        {
            double best_mean_square = -std::numeric_limits<double>::infinity();
            for (unsigned int channel_idx : selected_channels_)
            {
                if (ema_[channel_idx] > best_mean_square)
                {
                    best_mean_square = ema_[channel_idx];
                    current_channel_index_ = channel_idx;
                }
            }
        }

        channel_state_->set(current_channel_index_);

        if (packet_count_ % static_cast<int>(5 * fs_) == 0)
        {
            LOG(INFO) << name() << ". Packet " << packet_count_ + 1 << ": Selected channel " << current_channel_index_ + 1 << " (RMS: " << std::sqrt(ema_[current_channel_index_]) << "uV)";
        }

        idx_out = idx_out_port_->slot(0)->ClaimData(false);
        idx_out->set_data(current_channel_index_ + 1); // convert back to 1-based index
        idx_out->set_hardware_timestamp(data_in->hardware_timestamp());
        idx_out->set_source_timestamp(Clock::now());

        data_in_port_->slot(0)->ReleaseData();
        idx_out_port_->slot(0)->PublishData();

        packet_count_++;
    }
    LOG(INFO) << name() << " stopped working";
}

void ChannelSelector::Postprocess(ProcessingContext &context)
{
    LOG(INFO) << name() << ": Total messages processed: " << packet_count_ << ", last selected channel: " << current_channel_index_ + 1;
}

REGISTERPROCESSOR(ChannelSelector);
