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

class Producer : public IProcessor
{
public:
    Producer();

    void CreatePorts() override;
    void CompleteStreamInfo() override;
    void Preprocess(ProcessingContext &context) override;
    void Process(ProcessingContext &context) override;
    void Postprocess(ProcessingContext &context) override;

protected:
    // Data ports
    PortOut<MultiChannelType<float>> *data_out_port_;
    PortOut<MultiChannelType<double>> *meta_out_port_;

    // Options
    options::String path_{"run://"};
    options::Double fs_{10000.0};
    options::Double carrier_amplitude_{3.0};
    options::Double carrier_frequency_{8.0};
    options::String modulation_type_{"phase"};
    options::Double modulation_amplitude_{1.0};
    options::Double modulation_frequency_{0.1};
    options::Value<unsigned int, false> nchannels_{10};
    options::Value<unsigned int, false> nsamples_{1};
    options::Int n_messages_{-1};

    // Runtime state
    BroadcasterState<double> *iaf_state_ = nullptr;
    double current_iaf_ = 10.0;
    int packet_count_ = 0;
};
