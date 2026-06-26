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

#include <exception>
#include <string>

MultiChannelFilter::MultiChannelFilter() : IProcessor() {
  add_option("filter", filter_def_, "Filter definition.", true);
  add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
}

void MultiChannelFilter::Configure(const GlobalContext &context) {
  // Nothing to configure at this stage; filter construction happens in Prepare.
}

void MultiChannelFilter::CreatePorts() {
  data_in_port_ = create_input_port<MultiChannelType<float>>(
      "in", MultiChannelType<float>::Capabilities(ChannelRange(1, MAX_NCHANNELS)),
      PortInPolicy(SlotRange(0, MAX_NCHANNELS)));

  data_out_port_ = create_output_port<MultiChannelType<float>>(
      "out", MultiChannelType<float>::Parameters(), PortOutPolicy(SlotRange(0, MAX_NCHANNELS)));
}

void MultiChannelFilter::CompleteStreamInfo() {
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
  const auto &info = data_in_port_->streaminfo(0);
  const auto &p = info.parameters<MultiChannelType<float>::Parameters>();
  LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate;
  double fs = p.sample_rate;

  if (!filter_def_()["file"]) {
    std::string filter_name = filter_def_()["name"].as<std::string>();
    std::string filename = filter_name + "_" + std::to_string(fs) + ".txt";
    std::string f = context.resolve_path(filename, "filters");
    LOG(INFO) << name() << " Reading filter coefficients from: " << f;
    filter_template_.reset(dsp::filter::construct_from_file(f));
  } else {
    std::string f = context.resolve_path(filter_def_()["file"].as<std::string>(), "filters");
    filter_template_.reset(dsp::filter::construct_from_file(f));
  }

  // Instantiate one filter per input slot, sized to that slot's channel count
  filters_.clear();
  for (int k = 0; k < data_in_port_->number_of_slots(); ++k) {
    filters_.push_back(std::move(
        std::unique_ptr<dsp::filter::IFilter>(filter_template_->clone())));
    const auto &slot_info = data_in_port_->streaminfo(k);
    const auto &slot_params = slot_info.parameters<MultiChannelType<float>::Parameters>();
    filters_.back()->realize(slot_params.nchannels);
  }
}

void MultiChannelFilter::Preprocess(ProcessingContext &context) {
  packet_count_ = 0;
}

void MultiChannelFilter::Process(ProcessingContext &context) {
  MultiChannelType<float>::Data *data_in = nullptr;
  MultiChannelType<float>::Data *data_out = nullptr;
  auto nslots = data_in_port_->number_of_slots();
  decltype(nslots) k = 0;

  while (!context.terminated()) {
    if (n_messages_() != -1 && packet_count_ >= n_messages_()) {
      break;
    }

    for (k = 0; k < nslots; ++k) {
      if (!data_in_port_->slot(k)->RetrieveData(data_in)) {
        break;
      }

      data_out = data_out_port_->slot(k)->ClaimData(false);

      filters_[k]->process_by_channel(data_in->nsamples(), data_in->data(),
                                      data_out->data());

      data_out->set_sample_timestamps(data_in->sample_timestamps());
      data_out->CloneTimestamps(*data_in);

      data_out_port_->slot(k)->PublishData();
      data_in_port_->slot(k)->ReleaseData();
    }
    packet_count_++;
  }
  LOG(INFO) << name() << " stopped working";
}

void MultiChannelFilter::Postprocess(ProcessingContext &context) {
  LOG(INFO) << name() << ": Total messages processed: " << packet_count_;
}

REGISTERPROCESSOR(MultiChannelFilter)
