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

#include "PhaseEstimator.hpp"
#include "logging/log.hpp"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <limits>


PhaseEstimator::PhaseEstimator() : IProcessor(PRIORITY_HIGH) {
  add_option("n_messages", n_messages_, "Number of packets to receive");
}

void PhaseEstimator::CreatePorts() {
  data_in_port_ = create_input_port<MultiChannelType<float>>(
      "in", 
      MultiChannelType<float>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
      PortInPolicy(SlotRange(1)));

  data_out_port_ = create_output_port<MultiChannelType<float>>(
      "out",
      MultiChannelType<float>::Parameters(1,1,1),
      PortOutPolicy(SlotRange(1),200,WaitStrategy::kBlockingStrategy));
}

void PhaseEstimator::CompleteStreamInfo() {
  const auto &input_info = data_in_port_->slot(0)->streaminfo();
  const auto &input_params =
      input_info.parameters<MultiChannelType<float>::Parameters>();

  dynamic_cast<StreamInfo<MultiChannelType<float>>&>(
      data_out_port_->slot(0)->streaminfo())
      .set_parameters(input_params);
}

void PhaseEstimator::Preprocess(ProcessingContext &context) {
  printf("\n");
  const auto& info = data_in_port_->streaminfo(0);
  const auto& p = info.parameters<MultiChannelType<float>::Parameters>();
  printf("Stream parameters - nchannels: %lu, nsamples: %lu, sample_rate: %f\n", p.nchannels, p.nsamples, p.sample_rate);
  fs_ = p.sample_rate;
  sample_window.set_capacity(fs_ * (1.0/10.0)*2.0);  // 2 cycles of a 10 Hz sine wave
  printf("Sample window size set to %lu\n", sample_window.capacity());
}

void PhaseEstimator::Process(ProcessingContext &context) {
  MultiChannelType<float>::Data* data_in;
  MultiChannelType<float>::Data *data_out = nullptr;

  // Measurement phase
  while (!context.terminated()) {
    if (n_messages_() != -1 && packet_count_ >= n_messages_()) {
      break;
    }

    if (!data_in_port_->slot(0)->connected()) {
    // No upstream connected, wait for a short time before checking again
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    continue;
    }
  
    // Try to retrieve data
    if (!data_in_port_->slot(0)->RetrieveData(data_in)) {
      break;
    }

    data_out = data_out_port_->slot(0)->ClaimData(false);

    float sample = data_in->data_sample(0,0);  // Get the first sample of the first channel
    sample_window.push_back(sample);

    data_out = data_in;   // Echo the input data to output
    // data_out->data() = data_in->data();   
    // data_out->set_source_timestamp(data_in->source_timestamp());

    data_in_port_->slot(0)->ReleaseData();
    data_out_port_->slot(0)->PublishData();  

    packet_count_++;

  }


}

void PhaseEstimator::Postprocess(ProcessingContext &context) {
  printf("\n PhaseEstimator:Total messages processed: %d", packet_count_);
}

REGISTERPROCESSOR(PhaseEstimator);
