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
#include "dio/dio.hpp"
#include "eventdata/eventdata.hpp"
#include "iprocessor.hpp"
#include "utilities/time.hpp"

typedef std::map<std::string, std::map<std::string, std::vector<uint32_t>>>
    ProtocolYAMLMap;
typedef std::map<std::string, std::unique_ptr<DigitalOutputProtocol>>
    ProtocolMap;

class DigitalOutput : public IProcessor {

public:
  DigitalOutput();
  void CreatePorts() override;
  void Configure(const GlobalContext &context) override;
  void Process(ProcessingContext &context) override;

  // DATA PORTS
protected:
  PortIn<EventType> *data_in_port_;

protected:
  options::Measurement<double, false> initial_lockout_period_{
      300, "ms", options::positive<double>(true)};

  options::Measurement<unsigned int, false> pulse_width_{
      400, "ms", options::positive<double>(true)};

  options::String device_type_{};
  options::Value<std::uint32_t, false> nchannels_{16};
  options::Value<ProtocolYAMLMap, false> protocols_yaml_{};

  options::Bool event_log_{true};

  std::unique_ptr<DigitalDevice> device_;
  ProtocolMap protocols_;
};
