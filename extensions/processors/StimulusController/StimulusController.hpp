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
#include "scalardata/scalardata.hpp"
#include "multichanneldata/multichanneldata.hpp"
#include <complex>
#include <vector>

class StimulusController : public IProcessor {
  public:
    StimulusController();

  void CreatePorts() override;
  void CompleteStreamInfo() override;
  void Prepare(GlobalContext &context) override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;

  // VARIABLES
  protected:
    unsigned int packet_count_ = 0;
    double output_ = 0;
    double stim_onset_rad_ = 0;
    double stim_dur_rad_ = 0;

    const uint32_t MAX_NCHANNELS=384;

  // DATA PORTS
  protected:
    PortIn<MultiChannelType<float>> *data_in_port_;
    PortOut<ScalarType<float>> *data_out_port_;

  // OPTIONS
  protected:
    options::Value<int, false> n_messages_{-1};
    options::Value<float, false> stim_onset_deg{0};
    options::Value<float, false> stim_dur_deg{90};

};
