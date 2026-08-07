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
#include <readerwriterqueue/readerwriterqueue.h>
#include <alsa/asoundlib.h>
#include "miniaudio/miniaudio.c"
#include <atomic>
#include <complex>
#include <mutex>
#include <random>
#include <thread>
#include <cstdint>
#include <vector>

using namespace moodycamel;

class StimulusController : public IProcessor {
  public:
    StimulusController();

    void CreatePorts() override;
    void CompleteStreamInfo() override;
    void Prepare(GlobalContext &context) override;
    void Preprocess(ProcessingContext &context) override;
    void Unprepare(GlobalContext &context) override;
    void Process(ProcessingContext &context) override;
    void Postprocess(ProcessingContext &context) override;

  protected:
    // Data ports
    PortIn<MultiChannelType<double>>  *data_in_port_;
    PortOut<MultiChannelType<double>> *data_out_port_;

    // Options
    options::Int    n_messages_{-1};
    options::Double stim_onset_deg_{0};
    options::Double audio_latency_s_{0};
    options::Double erp_latency_s_{0};        // auditory evoked response potential latency (s)
    options::Bool   correct_latencies_{true};

    options::Double stim_period_ms_{0.5};
    options::Double stim_amplitude_{0.7};
    options::Int    stim_num_octaves_{16};      // Voss-McCartney pink noise octaves

    options::String stim_dur_unit_{"deg"};     // "deg" or "ms"
    options::Double stim_dur_ms_{20};          // fallback burst duration when IAF is unavailable
    options::Double stim_dur_deg_{90};

    options::String audio_device_{"hw:1,0"};
    options::Int    audio_sample_rate_{44100};
    options::Int    audio_channels_{2};
    options::String audio_format_{"s32"};      // "float", "s16", or "s32"

    options::Bool   randomize_stim_onset_{false};
    options::Double min_stim_dist_sec_{0};
    options::Double max_stim_dist_sec_{-1};
    options::Bool   use_background_sound_{false};
    options::Int    background_dB_{18};        // stimulus level above background sound (dB)

    // Runtime state — phase / burst logic
    unsigned int packet_count_ = 0;
    int stimuli_count_ = 0;
    bool output_      = false;
    bool last_output_ = false;
    double stim_onset_rad_ = 0;
    double stim_dur_rad_   = 0;

    // Cached enum for stim_dur_unit_ — avoids string comparison every packet in Process()
    enum class DurUnit { kMs, kDeg };
    DurUnit dur_unit_ = DurUnit::kMs;

    double fs_ = 0;          // input stream sample rate (Hz), set in CompleteStreamInfo
    FollowerState<double> *iaf_state_ = nullptr;

    // Audio burst playback
    virtual bool compute_burst_params_(double iaf);

    void build_audio_buffers_();
    void set_master_volume_(const std::string &card, long pct);
    bool start_audio_();
    snd_pcm_sframes_t write_with_recovery_(snd_pcm_t *pcm, const void *buf, snd_pcm_uframes_t frames);
    void stop_audio_() noexcept;
    void audio_thread_main_();
    bool set_hw_params_interleaved_(snd_pcm_t *pcm,
        int sample_rate,
        int channels,
        snd_pcm_uframes_t period_frames,
        snd_pcm_uframes_t buffer_frames,
        snd_pcm_format_t format,
        const char *format_name,
        std::string *fail_step,
        int *fail_rc);

    std::atomic<bool> audio_running_{false};
    std::atomic<bool> audio_trigger_pending_{false};
    std::atomic<bool> stim_enabled_{true};  // externally gated via "set_enabled" apply command
    YAML::Node SetEnabled(const YAML::Node &node);
    std::thread audio_thread_;
    std::mutex audio_mutex_;
    snd_pcm_t *pcm_ = nullptr;

    double period_ms_ = 0;
    double burst_precompute_ms_ = 1000; // pre-compute 1 second of audio buffers
    double burst_ms_ = 0;
    double last_iaf_ = std::numeric_limits<double>::quiet_NaN();

    double fs_audio_ = 0;
    int burst_frames_ = 0;
    int period_frames_ = 0;

    std::unique_ptr<moodycamel::ReaderWriterQueue<float>> background_sound_buffer_;
    float gain_ = 0.0;
    float smoothed_gain_ = 0.0;   // per-sample ramped gain, chases gain_/0 in the audio thread
    float bg_gain_ = 1.0;   // background scale factor, reduced when stimulus would cause clipping
    float power_s_ = 0.0;   // power of the stimulus signal (for gain normalization)
    std::vector<double> sound_buf_;
    bool valid_background_ = false;
    ma_decoder decoder_;

    snd_pcm_format_t pcm_format_ = SND_PCM_FORMAT_FLOAT_LE; // active PCM format (set in start_audio_)

    const uint32_t MAX_NCHANNELS = 384;
};