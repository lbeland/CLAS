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

#include "ChannelReduction.hpp"
#include "logging/log.hpp"
#include <limits>
#include <sstream>

ChannelReduction::ChannelReduction() : IProcessor(PRIORITY_HIGH)
{
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("default_channel_index", default_channel_index_, "Channel index to use when no valid index is received from channel_state (1-based).");

    channel_state_ = create_follower_state<unsigned int>(
        "ch_idx", default_channel_index_() - 1, Permission::NONE,
        "Current selected channel index (zero-based) shared by an upstream processor.");
}

void ChannelReduction::CreatePorts()
{
    data_in_port_ = create_input_port<MultiChannelType<double>>(
        "in",
        MultiChannelType<double>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
        PortInPolicy(SlotRange(0, MAX_NCHANNELS)));

    data_out_port_ = create_output_port<MultiChannelType<double>>(
        "out",
        MultiChannelType<double>::Parameters(1, 1, 1), // Placeholder, set in CompleteStreamInfo
        PortOutPolicy(SlotRange(0, MAX_NCHANNELS), 200, WaitStrategy::kBlockingStrategy));
}

void ChannelReduction::CompleteStreamInfo()
{
    const auto &input_params = data_in_port_->slot(0)->streaminfo().parameters<MultiChannelType<double>::Parameters>();

    // Pass through only the selected channel; keep nsamples and sample_rate from input
    data_out_port_->streaminfo(0).set_parameters(MultiChannelType<double>::Parameters(1, input_params.nsamples, input_params.sample_rate));
    data_out_port_->streaminfo(0).set_stream_rate(data_in_port_->streaminfo(0));
}

void ChannelReduction::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);
    const auto &p = info.parameters<MultiChannelType<double>::Parameters>();
    LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate;
}

void ChannelReduction::Preprocess(ProcessingContext &context)
{
    packet_count_ = 0;
    ch_idx_ = default_channel_index_() - 1; // convert to 0-based
}

void ChannelReduction::Process(ProcessingContext &context)
{
    MultiChannelType<double>::Data *data_in = nullptr;
    MultiChannelType<double>::Data *data_out = nullptr;

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

        ch_idx_ = channel_state_->get();

        data_out = data_out_port_->slot(0)->ClaimData(false);
        data_out->set_data_sample(0, 0, data_in->data_sample(0, ch_idx_));
        data_out->set_sample_timestamps(data_in->sample_timestamps());
        data_out->set_hardware_timestamp(data_in->hardware_timestamp());
        data_out->set_source_timestamp(Clock::now());

        data_in_port_->slot(0)->ReleaseData();
        data_out_port_->slot(0)->PublishData();

        packet_count_++;
    }
    LOG(INFO) << name() << " stopped working";
}

void ChannelReduction::Postprocess(ProcessingContext &context)
{
    LOG(INFO) << name() << ": Total messages processed: " << packet_count_;
}

REGISTERPROCESSOR(ChannelReduction);
