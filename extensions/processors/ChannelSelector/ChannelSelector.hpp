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
#include <dsp/filter.hpp>
#include <boost/circular_buffer.hpp>
#include <complex>
#include <fftw3.h>
#include <vector>

class ChannelSelector : public IProcessor {
  public:
    ChannelSelector();

  void CreatePorts() override;
  void CompleteStreamInfo() override;
  void Prepare(GlobalContext &context) override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;

  // VARIABLES
  protected:
    unsigned int packet_count_ = 0;
    double first_timestamp_ = 0.0;
    float fs_ = 0.0;
    inline static boost::circular_buffer<float> sample_window{1};  // Initialized with size 1, will be resized in Prepare

    const uint32_t MAX_NCHANNELS=384;

  // DATA PORTS
  protected:
    PortIn<MultiChannelType<float>> *data_in_port_;
    PortOut<MultiChannelType<float>> *data_out_port_;

  // OPTIONS
  protected:
    options::Value<int, false> n_messages_{-1};

    unsigned int current_channel_index_ = 0;
};
