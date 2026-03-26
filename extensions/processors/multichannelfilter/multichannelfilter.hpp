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

#include <memory>
#include <vector>

#include "iprocessor.hpp"
#include "multichanneldata/multichanneldata.hpp"
#include <dsp/filter.hpp>

class MultiChannelFilter : public IProcessor {
  // CONSTRUCTOR and OVERLOADED METHODS
 public:
  MultiChannelFilter();
  void Configure(const GlobalContext &context) override;
  void CreatePorts() override;
  void CompleteStreamInfo() override;
  void Prepare(GlobalContext &context) override;
  void Process(ProcessingContext &context) override;

  // VARIABLES
 protected:
  std::unique_ptr<dsp::filter::IFilter> filter_template_;
  std::vector<std::unique_ptr<dsp::filter::IFilter>> filters_;

  // DATA PORTS
 protected:
  PortIn<MultiChannelType<double>> *data_in_port_;
  PortOut<MultiChannelType<double>> *data_out_port_;

  // OPTIONS
 protected:
  options::Value<YAML::Node, false> filter_def_{};


 const uint32_t MAX_NCHANNELS=384;
};
