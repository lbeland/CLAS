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

class Consumer : public IProcessor {
 public:
    Consumer();

  void CreatePorts() override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;

 protected:
  PortIn<MultiChannelType<double>> *data_in_port_;

  options::Value<unsigned int, false> n_messages_{0};
  options::Value<std::string, false> output_file_{"Consumer.csv"};
  
  unsigned int packet_count_ = 0;
  unsigned int warm_up_count_ = 0;
  double first_timestamp_ = 0.0;
  inline static std::vector<double> recv_times;
  inline static std::vector<double> process_times;
};
