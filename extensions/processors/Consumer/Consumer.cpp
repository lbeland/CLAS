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

#include "Consumer.hpp"
#include "logging/log.hpp"
#include <chrono>
#include <sstream>

Consumer::Consumer() : IProcessor(PRIORITY_HIGH)
{
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("window_size", window_size_, "Number of packets to skip at startup before computing latency statistics.");
    add_option("path", path_, "Path (server-side) where to save data.");
}

void Consumer::CreatePorts()
{
    data_in_port_ = create_input_port<AnyType>(
        "in",
        AnyType::Capabilities(),
        PortInPolicy(SlotRange(0, MAX_NCHANNELS)));
}

void Consumer::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);

    if (info.datatype() == "MultiChannelType<float>")
    {
        try
        {
            const auto &p = info.parameters<MultiChannelType<float>::Parameters>();
            LOG(INFO) << name() << ": Data type:" << info.datatype() << ", nChannels: " << p.nchannels << ", nSamples: " << p.nsamples << ", Sample rate: " << p.sample_rate;
        }
        catch (const std::bad_any_cast &)
        {
            LOG(WARNING) << name() << ": Failed to cast to MultiChannelType<float>";
        }
    }
    else if (info.datatype() == "MultiChannelType<double>")
    {
        try
        {
            const auto &p = info.parameters<MultiChannelType<double>::Parameters>();
            LOG(INFO) << name() << ": Data type:" << info.datatype() << ", nChannels: " << p.nchannels << ", nSamples: " << p.nsamples << ", Sample rate: " << p.sample_rate;
        }
        catch (const std::bad_any_cast &)
        {
            LOG(WARNING) << name() << ": Failed to cast to MultiChannelType<double>";
        }
    }
    else if (info.datatype() == "scalar")
    {
        try
        {
            const auto &p = info.parameters<ScalarType<double>::Parameters>();
            LOG(INFO) << name() << ": Data type:" << info.datatype() << ", Default value: " << p.default_value;
        }
        catch (const std::bad_any_cast &)
        {
            LOG(WARNING) << name() << ": Failed to cast to ScalarType";
        }
    }
}

void Consumer::Preprocess(ProcessingContext &context)
{
    packet_count_ = 0;
}

void Consumer::Process(ProcessingContext &context)
{
    AnyType::Data *data_in = nullptr;
    TimePoint receive_timestamp;
    TimePoint source_timestamp;
    double latency_ms = 0.0;

    while (!context.terminated())
    {
        if (n_messages_() != -1 && packet_count_ >= n_messages_())
        {
            break;
        }

        for (int slot_idx = 0; slot_idx < data_in_port_->number_of_slots(); slot_idx++)
        {
            if (!data_in_port_->slot(slot_idx)->RetrieveData(data_in))
            {
                break;
            }
            receive_timestamp = Clock::now();
            source_timestamp = data_in->source_timestamp();
            data_in_port_->slot(slot_idx)->ReleaseData();
        }

        latency_ms = std::chrono::duration<double, std::milli>(receive_timestamp - source_timestamp).count();
        if (latency_ms > max_latency_)
        {
            max_latency_ = latency_ms;
            max_latency_index_ = packet_count_;
        }
        if (packet_count_ >= window_size_())
        {
            int latency_index = packet_count_ - window_size_();
            if (latency_index == 0)
            {
                mean_latency_ = latency_ms;
            }
            else
            {
                mean_latency_ = ((latency_index)*mean_latency_ + latency_ms) / (latency_index + 1);
            }
        }
        packet_count_++;
    }
    LOG(INFO) << name() << " stopped working";
}

void Consumer::Postprocess(ProcessingContext &context)
{
    std::ostringstream statistic_print;
    statistic_print << "\n ---------------- \n " << name() << ": Total messages processed: " << packet_count_;

    if (packet_count_ == 0)
    {
        return;
    }
    statistic_print << "\n Max latency (ms): " << max_latency_ << " (idx:" << max_latency_index_ << ")";
    statistic_print << "\n Mean latency (ms): " << mean_latency_ << "\n";

    std::cout << statistic_print.str() << "\n";
}

REGISTERPROCESSOR(Consumer);
