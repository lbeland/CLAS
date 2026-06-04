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

#include "StimulusControllerRand.hpp"
#include "utilities/time.hpp"
#include "logging/log.hpp"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <sstream>
#include <string>
#include <complex>
#include <cstdint>
#include <random>

StimulusControllerRand::StimulusControllerRand() : StimulusController()
{
    add_option("min_stim_dist_sec", min_stim_dist_sec_, "Minimum distance between stimuli in seconds.");
    add_option("max_stim_dist_sec", max_stim_dist_sec_, "Maximum distance between stimuli in seconds.");
}

void StimulusControllerRand::CreatePorts()
{
    // No inport or output ports
}

void StimulusControllerRand::CompleteStreamInfo()
{
    // No inport or output ports
}

bool StimulusControllerRand::compute_burst_params_()
{
    const int sample_rate = std::max(1, audio_sample_rate_());
    int new_burst_frames = 0;

    // Fixed duration
    const int burst_ms = std::max(1, stim_dur_ms_());
    new_burst_frames = burst_ms * sample_rate / 1000;

    LOG(INFO) << name() << "Stimulus duration: " << burst_ms << " ms, " << new_burst_frames << " frames at " << sample_rate << " Hz";
    new_burst_frames = std::max(1, new_burst_frames);
    const bool changed = (new_burst_frames != burst_frames_);
    burst_frames_ = new_burst_frames;
    period_ms_ = stim_period_ms_(); // silence period
    return changed;
}

void StimulusControllerRand::Prepare(GlobalContext &context)
{
    if (min_stim_dist_sec_() <= stim_dur_ms_() / 1000.0)
    {
        throw std::runtime_error("min_stim_dist_sec must be greater than stim_dur_ms");
    }

    compute_burst_params_();
    build_audio_buffers_(); // uses burst_frames_ and period_ms_

}

void StimulusControllerRand::Preprocess(ProcessingContext &context)
{
    stimuli_count_ = 0;
    if (!start_audio_())
    {
        LOG(ERROR) << name() << " failed to start audio playback (device: " << audio_device_() << ")";
        throw std::runtime_error("Failed to start audio playback");
    }
}

void StimulusControllerRand::Process(ProcessingContext &context)
{
    TimePoint last_stim_time_ = Clock::now();
    std::random_device rd;   // non-deterministic generator
    std::mt19937 gen(rd());  // to seed mersenne twister.
                        // replace the call to rd() with a
                        // constant value to get repeatable
                        // results.
    std::uniform_real_distribution<double> distrib(min_stim_dist_sec_(), max_stim_dist_sec_());

    double stim_dist_sec_ = distrib(gen);

    while (!context.terminated())
    {

        TimePoint now = Clock::now();
        double time_since_last_stim = std::chrono::duration<double>(now - last_stim_time_).count();

        if (time_since_last_stim >= stim_dist_sec_)
        {
             output_ = true;
        }
        else
        {
             output_ = false;
        }

        if (output_ && !last_output_)
        {
            audio_trigger_pending_.store(true);
            stimuli_count_++;
            last_stim_time_ = now;
            stim_dist_sec_ = distrib(gen);
        }
        last_output_ = output_;

        custom_sleep_for(uint64_t(5000)); // Sleep for 5 ms to prevent busy waiting (can freeze the whole system)

    }
}

void StimulusControllerRand::Postprocess(ProcessingContext &context)
{
    stop_audio_();
    LOG(INFO) << name() << " Total stimuli sent: " << stimuli_count_;
}

REGISTERPROCESSOR(StimulusControllerRand);