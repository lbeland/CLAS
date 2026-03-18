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
  printf("Stream parameters - nchannels: %u, nsamples: %u, sample_rate: %f\n", p.nchannels, p.nsamples, p.sample_rate);
}

void PhaseEstimator::Process(ProcessingContext &context) {
  MultiChannelType<float>::Data* data_in;
  MultiChannelType<float>::Data *data_out = nullptr;

  // Measurement phase
  while (!context.terminated()) {
    
    // Try to retrieve data
    if (!data_in_port_->slot(0)->RetrieveData(data_in)) {
      break;
    }

    // double timestamp = data_in->source_timestamp().time_since_epoch().count();
    
    packet_count_++;
    auto sample = data_in->data(); 
    auto timestamp = data_in->source_timestamp();

    data_in_port_->slot(0)->ReleaseData();

    data_out = data_out_port_->slot(0)->ClaimData(false);

    // data_out->data()[0] = data_in->data()[0];  // Echo the input data to output
    // std::fill(data_out->data().begin(), data_out->data().end(), data_in->data()[0]);
    data_out->data() = sample;  // Echo the input data to output
    data_out->set_source_timestamp(timestamp);
    // printf("%s. Received and sent packet %u with sample %f\n", name().c_str(), packet_count_, sample);

    data_out_port_->slot(0)->PublishData();  

    // custom_sleep_for(500000);
  }


}

void PhaseEstimator::Postprocess(ProcessingContext &context) {
  printf("\n PhaseEstimator:Total messages processed: %d", packet_count_);
}

REGISTERPROCESSOR(PhaseEstimator);
