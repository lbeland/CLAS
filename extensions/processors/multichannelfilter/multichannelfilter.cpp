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

#include "multichannelfilter.hpp"

#include <chrono>
#include <exception>
#include <string>
#include <thread>

MultiChannelFilter::MultiChannelFilter() : IProcessor() {
  add_option("filter", filter_def_, "Filter definition.", true);
}

void MultiChannelFilter::Configure(const GlobalContext &context) {
  if (!filter_def_()["file"]) {
    filter_template_.reset(dsp::filter::construct_from_yaml(filter_def_()));
  } else {
    std::string f = context.resolve_path(
        filter_def_()["file"].as<std::string>(), "filters");
    filter_template_.reset(dsp::filter::construct_from_file(f));
  }
}

void MultiChannelFilter::CreatePorts() {
  data_in_port_ = create_input_port<MultiChannelType<double>>(
      "data", MultiChannelType<double>::Capabilities(ChannelRange(1, MAX_NCHANNELS)),
      PortInPolicy(SlotRange(0, MAX_NCHANNELS)));

  data_out_port_ = create_output_port<MultiChannelType<double>>(
      "data", MultiChannelType<double>::Parameters(), PortOutPolicy(SlotRange(0, MAX_NCHANNELS)));
}

void MultiChannelFilter::CompleteStreamInfo() {
  // check if we have the same number of input and output slots
  if (data_in_port_->number_of_slots() != data_out_port_->number_of_slots()) {
    auto err_msg = "Number of output slots (" +
                   std::to_string(data_out_port_->number_of_slots()) +
                   ") on port '" + data_out_port_->name() +
                   "' does not match number of input slots (" +
                   std::to_string(data_in_port_->number_of_slots()) +
                   ") on port '" + data_in_port_->name() + "'.";
    throw ProcessingStreamInfoError(err_msg, name());
  }

  for (int k = 0; k < data_in_port_->number_of_slots(); ++k) {
    data_out_port_->streaminfo(k).set_stream_rate(
        data_in_port_->streaminfo(k).stream_rate());
    const auto &input_info = data_in_port_->slot(k)->streaminfo();
    const auto &input_params = input_info.parameters<MultiChannelType<float>::Parameters>();
    data_out_port_->streaminfo(k).set_parameters(input_params);
  }
}

void MultiChannelFilter::Prepare(GlobalContext &context) {
  // realize filter for each input slot, dependent on the number of channels
  // upstream is sending
  filters_.clear();
  for (int k = 0; k < data_in_port_->number_of_slots(); ++k) {
    filters_.push_back(std::move(
        std::unique_ptr<dsp::filter::IFilter>(filter_template_->clone())));
    const auto& info = data_in_port_->streaminfo(0);
    const auto& p = info.parameters<MultiChannelType<float>::Parameters>();
    filters_.back()->realize(p.nchannels);
  }
}

void MultiChannelFilter::Process(ProcessingContext &context) {
  MultiChannelType<double>::Data *data_in = nullptr;
  MultiChannelType<double>::Data *data_out = nullptr;
  auto nslots = data_in_port_->number_of_slots();
  decltype(nslots) k = 0;

  while (!context.terminated()) {
    // go through all slots
    for (k = 0; k < nslots; ++k) {
      // retrieve new data
      if (!data_in_port_->slot(k)->RetrieveData(data_in)) {
        break;
      }

      // claim output data buckets
      data_out = data_out_port_->slot(k)->ClaimData(false);

      // filter incoming data
      filters_[k]->process_by_channel(data_in->nsamples(), data_in->data(),
                                      data_out->data());

      data_out->set_sample_timestamps(data_in->sample_timestamps());
      data_out->CloneTimestamps(*data_in);

      // publish and release data
      data_out_port_->slot(k)->PublishData();
      data_in_port_->slot(k)->ReleaseData();
    }
  }
}

REGISTERPROCESSOR(MultiChannelFilter)
