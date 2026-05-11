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

#include "StimulusController.hpp"
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
#include <complex>
namespace
{

} // namespace

StimulusController::StimulusController() : IProcessor(PRIORITY_HIGH)
{
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("stim_onset_deg", stim_onset_deg, "Stimulus onset phase in degrees.");
    add_option("stim_dur_deg", stim_dur_deg, "Stimulus duration in degrees.");

}

void StimulusController::CreatePorts()
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

void StimulusController::CompleteStreamInfo()
{
    // Set the parameters for the output stream
    for (int k = 0; k < data_out_port_->number_of_slots(); ++k)
    {
        data_out_port_->streaminfo(k).set_stream_rate(data_in_port_->streaminfo(0).stream_rate());
    }
}

void StimulusController::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);
    const auto &p = info.parameters<MultiChannelType<float>::Parameters>();
    LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate << "\n";

    stim_onset_rad_ = stim_onset_deg() * (1.0f / 180.0f * std::acos(-1.0f));
    stim_dur_rad_ = stim_dur_deg() * (1.0f / 180.0f * std::acos(-1.0f));
}


void StimulusController::Process(ProcessingContext &context)
{
    MultiChannelType<float>::Data *data_in;
    ScalarType<float>::Data *data_out;

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
        data_out->CloneTimestamps(*data_in);

        data_in_port_->slot(0)->ReleaseData();
        double diff = data_in->data_sample(0,0) - stim_onset_rad_;
        if (std::abs(std::atan2(std::sin(diff), std::cos(diff))) < stim_dur_rad_/2) //(1/180.0f * std::acos(-1.0f)): 1 degree in radians
        {
            output_ = 1;
            // LOG(INFO) << name() << ". Packet " << packet_count_ + 1 << ": Phase " << data_in->data_sample(0,0) << ", Atan2:" << std::atan2(std::sin(diff), std::cos(diff)) << ", output: " << output_;
        }
        else 
        {
            output_ = 0;
        }
        
        data_out->set_data(output_);
        data_out_port_->slot(0)->PublishData();

        packet_count_++;
    }
}

void StimulusController::Postprocess(ProcessingContext &context)
{
    printf("\n ---------------- \n StimulusController: Total messages processed: %d", packet_count_);
}

// void StimulusController::Unprepare(GlobalContext &context)
// {
//     // Save FFTW wisdom for future runs to speed up plan creation
//     fftwf_export_wisdom_to_filename(context.resolve_path("fftw_wisdom.txt", "fft_wisdom").c_str());
// }

REGISTERPROCESSOR(StimulusController);
