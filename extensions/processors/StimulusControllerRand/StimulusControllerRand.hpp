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
#include "StimulusController/StimulusController.hpp"
#include <alsa/asoundlib.h>
#include <atomic>
#include <complex>
#include <mutex>
#include <random>
#include <thread>
#include <cstdint>
#include <vector>

class StimulusControllerRand : public StimulusController
{
public:
    StimulusControllerRand();

    void CreatePorts() override;
    void CompleteStreamInfo() override;
    void Prepare(GlobalContext &context) override;
    void Process(ProcessingContext &context) override;
    void Postprocess(ProcessingContext &context) override;

    // VARIABLES
protected:
    bool compute_burst_params_();

    int stimuli_count_ = 0;

    // OPTIONS
protected:
    options::Double min_stim_dist_sec_{1};
    options::Double max_stim_dist_sec_{10};
};