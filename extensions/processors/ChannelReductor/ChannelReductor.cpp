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

#include "ChannelReductor.hpp"
#include "logging/log.hpp"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <limits>
#include <sstream>

ChannelReductor::ChannelReductor() : IProcessor(PRIORITY_HIGH)
{
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("default_channel_index", default_channel_index_, "Default channel index to select if no valid index is received from channel_state (zero-based).");
    
    channel_state_ = create_follower_state<unsigned int>(
            "ch_idx", default_channel_index_()-1, Permission::NONE,
            "Current selected channel index (zero-based) shared by an upstream processor.");

}

void ChannelReductor::CreatePorts()
{
    data_in_port_ = create_input_port<MultiChannelType<float>>(
        "in",
        MultiChannelType<float>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)), // Accept only one channel
        PortInPolicy(SlotRange(0, MAX_NCHANNELS)));

    data_out_port_ = create_output_port<MultiChannelType<float>>(
        "out",
        MultiChannelType<float>::Parameters(1, 1, 1), // Placeholder, will be set in CompleteStreamInfo
        PortOutPolicy(SlotRange(0, MAX_NCHANNELS), 200, WaitStrategy::kBlockingStrategy));
}

void ChannelReductor::CompleteStreamInfo()
{
    const auto &input_params = data_in_port_->slot(0)->streaminfo().parameters<MultiChannelType<float>::Parameters>();

    // only pass through the selected channel, so set output nchannels to 1 but keep nsamples and sample_rate the same as input
    data_out_port_->streaminfo(0).set_parameters(MultiChannelType<float>::Parameters(1, input_params.nsamples, input_params.sample_rate));
    data_out_port_->streaminfo(0).set_stream_rate(data_in_port_->streaminfo(0));
}

void ChannelReductor::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);
    const auto &p = info.parameters<MultiChannelType<float>::Parameters>();
    LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate << "\n";
    
}

void ChannelReductor::Preprocess(ProcessingContext &context)
{
    packet_count_ = 0;
    ch_idx = default_channel_index_() - 1;
}


void ChannelReductor::Process(ProcessingContext &context)
{
    MultiChannelType<float>::Data *data_in = nullptr;
    MultiChannelType<float>::Data *data_out = nullptr;

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

        ch_idx = channel_state_->get();

        // Claim output buffer
        data_out = data_out_port_->slot(0)->ClaimData(false);

        data_out->set_data_sample(0, 0, data_in->data_sample(0, ch_idx));
        data_out->set_sample_timestamps(data_in->sample_timestamps());
        data_out->CloneTimestamps(*data_in);

        data_in_port_->slot(0)->ReleaseData();
        data_out_port_->slot(0)->PublishData();

        packet_count_++;
    }
}

void ChannelReductor::Postprocess(ProcessingContext &context)
{
    std::ostringstream statistic_print;
    statistic_print << "\n ---------------- \n ChannelReductor: Total messages processed: " << packet_count_;
}

REGISTERPROCESSOR(ChannelReductor);
