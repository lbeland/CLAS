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
#pragma once

#include "iprocessor.hpp"
#include "multichanneldata/multichanneldata.hpp"
#include "options/options.hpp"
#include "utilities/time.hpp"

class ReplaySourceClient : public IProcessor {
 public:
  ReplaySourceClient();

  void CreatePorts() override;
  void CompleteStreamInfo() override;
  void Prepare(GlobalContext &context) override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;

 protected:
  PortOut<MultiChannelType<float>> *data_out_port_;

  // Replay source selection.
  options::String path_{"run://"};
  options::String file_{""};
  options::Int slot_{0};

  // Playback controls.
  options::Bool loop_{false};
  options::Bool real_time_{true};
  options::Double speed_factor_{1.0};
  options::Int n_messages_{-1};

  // Output contract.
  options::Value<unsigned int, false> nchannels_{32};
  options::Value<unsigned int, false> nsamples_{1};
  options::Double fs_{1000.0};

  inline static std::vector<TimePoint> send_times;
};