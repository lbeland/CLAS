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
#include <chrono>
#include "utilities/time.hpp"

class Consumer : public IProcessor {
  public:
    Consumer();

    void CreatePorts() override;
    void Prepare(GlobalContext &context) override;
    void Preprocess(ProcessingContext &context) override;
    void Process(ProcessingContext &context) override;
    void Postprocess(ProcessingContext &context) override;

  protected:
    // Data ports
    PortIn<AnyType> *data_in_port_;

    // Options
    options::Int n_messages_{-1};
    options::Int window_size_{2000}; // packets to skip at startup before computing statistics
    options::String path_{"run://"};

    // Runtime state
    unsigned int packet_count_ = 0;
    double max_latency_ = 0.0;
    unsigned int max_latency_index_ = 0;
    double mean_latency_ = 0.0;

    const uint32_t MAX_NCHANNELS = 384;
};
