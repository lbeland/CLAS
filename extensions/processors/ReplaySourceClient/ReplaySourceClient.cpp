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

#include "ReplaySourceClient.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>

#include "logging/log.hpp"

ReplaySourceClient::ReplaySourceClient() : IProcessor(PRIORITY_HIGH) {
  add_option("path", path_, "Path (server-side) that contains recorded results.");
  add_option("file", file_, "Optional file path to a specific recorded EEG stream.");
  add_option("slot", slot_, "Recorded slot to replay when a processor wrote multiple streams.");
  add_option("loop", loop_, "Restart replay from the beginning after the last packet.");
  add_option("real_time", real_time_, "Replay using recorded timing instead of emitting as fast as possible.");
  add_option("speed_factor", speed_factor_, "Playback speed multiplier.");
  add_option("n_messages", n_messages_, "Number of packets to replay (-1 = all available packets).");
  add_option("nchannels", nchannels_, "Number of EEG channels to replay.");
  add_option("nsamples", nsamples_, "Number of samples per packet.");
  add_option("fs", fs_, "Replay sample frequency.");
}

void ReplaySourceClient::CreatePorts() {
  data_out_port_ = create_output_port<MultiChannelType<float>>(
      "out",
      MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()),
      PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));
}

void ReplaySourceClient::CompleteStreamInfo() {
  data_out_port_->slot(0)->streaminfo().set_parameters(
      MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()));
  data_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());
}

void ReplaySourceClient::Prepare(GlobalContext &context) {
  send_times.clear();

  LOG(INFO) << name() << " configured for replay."
            << " path=" << path_()
            << " file=" << file_()
            << " slot=" << slot_()
            << " nchannels=" << nchannels_()
            << " nsamples=" << nsamples_()
            << " fs=" << fs_()
            << " real_time=" << real_time_()
            << " speed_factor=" << speed_factor_()
            << " loop=" << loop_();
}

void ReplaySourceClient::Process(ProcessingContext &context) {
  // Draft scaffold: file-backed replay logic belongs here.
  // The intended behavior is to load a recorded EEG stream, emit the 32 EEG
  // channels on the output port, and optionally preserve timing.
  if (!context.terminated()) {
    LOG(WARNING) << name() << ": replay processing is a draft scaffold and does not emit data yet.";
  }
}

void ReplaySourceClient::Postprocess(ProcessingContext &context) {
  std::ostringstream statistic_print;
  statistic_print << "\n ---------------- \n Total replay packets emitted: " << send_times.size();
  std::cout << statistic_print.str();
}

REGISTERPROCESSOR(ReplaySourceClient)