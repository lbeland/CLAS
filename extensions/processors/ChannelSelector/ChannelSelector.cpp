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
#include <limits>
#include <cmath>
#include <string>


ChannelSelector::ChannelSelector() : IProcessor(PRIORITY_HIGH) {
  add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
  add_option("rms_window_seconds", rms_window_seconds_, "Length of the weighted RMS window in seconds (recent samples get higher weights).");

}

void ChannelSelector::CreatePorts() {
  data_in_port_ = create_input_port<MultiChannelType<float>>(
      "in", 
      MultiChannelType<float>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
      PortInPolicy(SlotRange(0,MAX_NCHANNELS)));

  data_out_port_ = create_output_port<MultiChannelType<float>>(
      "out",
      MultiChannelType<float>::Parameters(1,1,1), // Placeholder, will be set in CompleteStreamInfo
      PortOutPolicy(SlotRange(0,MAX_NCHANNELS),200,WaitStrategy::kBlockingStrategy));
}

void ChannelSelector::CompleteStreamInfo() {
  const auto &input_params = data_in_port_->slot(0)->streaminfo().parameters<MultiChannelType<float>::Parameters>();

  for (int k = 0; k < data_in_port_->number_of_slots(); ++k) {
    // only pass through the selected channel, so set output nchannels to 1 but keep nsamples and sample_rate the same as input
    data_out_port_->streaminfo(k).set_parameters(MultiChannelType<float>::Parameters(1, input_params.nsamples, input_params.sample_rate));
    data_out_port_->streaminfo(k).set_stream_rate(data_in_port_->streaminfo(k));
  }
}

void ChannelSelector::Prepare(GlobalContext &context) {
  const auto& info = data_in_port_->streaminfo(0);
  const auto& p = info.parameters<MultiChannelType<float>::Parameters>();
  LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate << "\n";

  fs_ = p.sample_rate;
  current_channel_index_ = 0;
  n_channels_ = p.nchannels;

  const double tau_seconds = rms_window_seconds_();
  rms_alpha_ = 1.0 - std::exp(-1.0 / (p.sample_rate * tau_seconds));
  rms_.assign(p.nchannels, 0.0);

  LOG(INFO) << name() << " RMS selector EMA tau: " << rms_window_seconds_()
            << " s, alpha: " << rms_alpha_ << ".";
}

void ChannelSelector::Process(ProcessingContext &context) {
  MultiChannelType<float>::Data *data_in = nullptr;
  MultiChannelType<float>::Data *data_out = nullptr;

  // Measurement phase
  while (!context.terminated()) {

    if (n_messages_() != -1 && packet_count_ >= n_messages_()) {
      break;
    }
  
    // Try to retrieve data
    if (!data_in_port_->slot(0)->RetrieveData(data_in)) {
      break;
    }

    double best_mean_square = -std::numeric_limits<double>::infinity();

    for (std::size_t channel_idx = 0; channel_idx < n_channels_; ++channel_idx) {
      const double sample = static_cast<double>(data_in->data_sample(0, channel_idx));
      const double mean_square = (1.0 - rms_alpha_) * rms_[channel_idx] + rms_alpha_ * (sample * sample);
      rms_[channel_idx] = mean_square;
      if (mean_square > best_mean_square) {
        best_mean_square = mean_square;
        current_channel_index_ = channel_idx;
      }
    }

    if (packet_count_ % int(fs_) == 0) {
      LOG(INFO) << name() << ". Packet " << packet_count_ + 1 << ": Selected channel " << current_channel_index_ << " (RMS: " << rms_[current_channel_index_] << ")";
    }

    // Claim output buffer
    data_out = data_out_port_->slot(0)->ClaimData(false);

    data_out->set_data_sample(0, 0, data_in->data_sample(0, current_channel_index_));

    data_out->CloneTimestamps(*data_in);

    data_in_port_->slot(0)->ReleaseData();
    data_out_port_->slot(0)->PublishData();

    packet_count_ ++;
      
  }

}

void ChannelSelector::Postprocess(ProcessingContext &context) {
  printf("\n ---------------- \n ChannelSelector: Total messages processed: %d, last selected channel: %u",
         packet_count_, current_channel_index_);
}

REGISTERPROCESSOR(ChannelSelector);
