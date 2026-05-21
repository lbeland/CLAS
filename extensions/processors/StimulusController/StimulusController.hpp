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
#include <alsa/asoundlib.h>
#include <atomic>
#include <complex>
#include <mutex>
#include <random>
#include <thread>
#include <cstdint>
#include <vector>

class StimulusController : public IProcessor {
  public:
    StimulusController();

  void CreatePorts() override;
  void CompleteStreamInfo() override;
  void Prepare(GlobalContext &context) override;
  void Unprepare(GlobalContext &context) override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;

  // VARIABLES
  protected:
    unsigned int packet_count_ = 0;
    bool output_      = false;
    bool last_output_ = false;
    double stim_onset_rad_ = 0;
    double stim_dur_rad_   = 0;

    // Cached enum for stim_dur_unit_ — avoids string comparison every packet in Process().
    enum class DurUnit { kMs, kDeg };
    DurUnit dur_unit_ = DurUnit::kMs;

    const uint32_t MAX_NCHANNELS=384;

    double fs_ = 0; // sample rate (Hz), set during CompleteStreamInfo()
    FollowerState<double>* iaf_state_ = nullptr;

    // Audio burst playback

    // Single source of truth: derives stim_dur_rad_ and burst_frames_ from
    // iaf (only used when stim_dur_unit_ == "deg"; ignored for "ms").
    // Returns true if burst_frames_ changed and buffers need rebuilding.
    bool compute_burst_params_(double iaf);

    void build_audio_buffers_();
    bool start_audio_();
    void stop_audio_() noexcept;
    void audio_thread_main_();

    std::atomic<bool> audio_running_{false};
    std::atomic<bool> audio_trigger_pending_{false};
    std::thread audio_thread_;
    std::mutex audio_mutex_;
    snd_pcm_t* pcm_ = nullptr;

    double period_ms_ = 0;
    double last_iaf_ = std::numeric_limits<double>::quiet_NaN(); // last IAF used to build buffers

    int burst_frames_ = 0;
    int period_frames_ = 0;
    std::vector<double> burst_buf_;
    std::vector<double> silence_buf_;

    // Optional integer buffers for direct hw devices
    std::vector<int16_t> burst_buf_s16_;
    std::vector<int16_t> silence_buf_s16_;

    // Active PCM format (set during start_audio_)
    snd_pcm_format_t pcm_format_ = SND_PCM_FORMAT_FLOAT_LE;

  // DATA PORTS
  protected:
    PortIn<MultiChannelType<double>> *data_in_port_;
    PortOut<MultiChannelType<double>> *data_out_port_;

  // OPTIONS
  protected:
    options::Value<int, false> n_messages_{-1};
    options::Value<double, false> stim_onset_deg_{0};
    options::Value<double, false> audio_latency_{0};
    options::Value<double, false> erp_latency_{0};  //auditory evoked response potential latency in seconds
    options::Value<int, false> stim_period_ms_{5};
    options::Value<double, false> stim_amplitude_{0.7};
    options::Value<int, false> stim_num_octaves_{6};

    options::Value<std::string, false> stim_dur_unit_{"deg"};
    options::Value<int, false> stim_dur_ms_{20};
    options::Value<double, false> stim_dur_deg_{90};

    // Audio options
    options::Value<std::string, false> audio_device_{"hw:1,0"};
    options::Value<int, false> audio_sample_rate_{44100};
    options::Value<int, false> audio_channels_{2};


    // Audio sample format: "float" (FLOAT_LE) or "s16" (S16_LE)
    options::Value<std::string, false> audio_format_{"float"};

};