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
#include "scalardata/scalardata.hpp"
#include <vector>

class ChannelSelector : public IProcessor {
  public:
    ChannelSelector();

  void CreatePorts() override;
  void CompleteStreamInfo() override;
  void Prepare(GlobalContext &context) override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;

  BroadcasterState<unsigned int>* channel_state_ = nullptr;

  // VARIABLES
  protected:
    unsigned int packet_count_ = 0;
    double fs_ = 0;

    const uint32_t MAX_NCHANNELS=384;

  // DATA PORTS
  protected:
    PortIn<MultiChannelType<float>> *data_in_port_;
    PortOut<MultiChannelType<float>> *data_out_port_;
    PortOut<ScalarType<unsigned int>> *idx_out_port;

  // OPTIONS
  protected:
    options::Int n_messages_{-1};
    options::Vector<int, false> channel_indices_;
    options::Double rms_window_seconds_{5.0};

    unsigned int current_channel_index_ = 0;
    double ema_mu_ = 1.0;
    unsigned int n_channels_ = 0;
    std::vector<unsigned int> selected_channels_;
    std::vector<double> ema_;
};
