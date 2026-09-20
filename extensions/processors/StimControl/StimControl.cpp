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

#include "StimControl.hpp"
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
#include <dirent.h>
#include <signal.h>
#include <unistd.h>
#include <cstdio>
#include <thread>

namespace
{
// Finds processes holding the ALSA PCM device node (parsed from "hw:C,D") open
// and sends them SIGTERM. Returns the number of processes signalled.
int sigterm_alsa_device_holders_(const std::string &dev)
{
    int card = -1, device = 0;
    if (sscanf(dev.c_str(), "hw:%d,%d", &card, &device) < 1 || card < 0)
        return 0;

    char snd_path[64];
    snprintf(snd_path, sizeof(snd_path), "/dev/snd/pcmC%dD%dp", card, device);

    int count = 0;
    DIR *proc = opendir("/proc");
    if (!proc) return 0;

    struct dirent *pent;
    while ((pent = readdir(proc)) != nullptr)
    {
        pid_t pid = static_cast<pid_t>(atoi(pent->d_name));
        if (pid <= 1) continue;

        char fd_dir[32];
        snprintf(fd_dir, sizeof(fd_dir), "/proc/%d/fd", pid);
        DIR *fds = opendir(fd_dir);
        if (!fds) continue;

        struct dirent *fent;
        while ((fent = readdir(fds)) != nullptr)
        {
            char link[64], target[256];
            snprintf(link, sizeof(link), "/proc/%d/fd/%s", pid, fent->d_name);
            ssize_t n = readlink(link, target, sizeof(target) - 1);
            if (n <= 0) continue;
            target[n] = '\0';
            if (strcmp(target, snd_path) == 0)
            {
                kill(pid, SIGTERM);
                ++count;
                break;
            }
        }
        closedir(fds);
    }
    closedir(proc);
    return count;
}

[[gnu::always_inline]] inline int16_t float_to_s16_(float v) {
    v = std::clamp(v, -1.0f, 1.0f);
    return static_cast<int16_t>(v * 32767.0f + 0.5f);
}

[[gnu::always_inline]] inline int32_t float_to_s32(float v) {
    v = std::clamp(v, -1.0f, 1.0f);
    return static_cast<int32_t>(v * 8388607.0f) << 8;
}
}

StimControl::StimControl() : IProcessor(PRIORITY_HIGH)
{
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("max_n_stimuli", max_n_stimuli_, "Maximum number of stimuli to present (-1 = infinite). Graph is stopped automatically");
    add_option("stim_onset_deg", stim_onset_deg_, "Stimulus onset phase in degrees.");
    add_option("audio_latency_s", audio_latency_s_, "Estimated audio latency in seconds (for phase correction).");
    add_option("erp_latency_s", erp_latency_s_, "Estimated auditory evoked response potential latency in seconds (for phase correction).");
    add_option("correct_latencies", correct_latencies_, "Whether to apply latency corrections to stimulus timing (default: true).");

    add_option("audio_device", audio_device_, "ALSA device string for playback (e.g. hw:1,0 or default).");
    add_option("audio_sample_rate", audio_sample_rate_, "Audio sample rate (Hz).");
    add_option("audio_channels", audio_channels_, "Number of output channels.");
    add_option("stim_period_ms", stim_period_ms_, "Silence period chunk size in milliseconds.");
    add_option("stim_amplitude", stim_amplitude_, "Burst amplitude (0..1).");
    add_option("stim_num_octaves", stim_num_octaves_, "Voss-McCartney pink noise octaves.");
    add_option("audio_format", audio_format_, "Sample format: float (FLOAT_LE) or s16 (S16_LE). hw:* devices often require s16.");

    add_option("stim_dur_deg", stim_dur_deg_, "Stimulus duration in degrees.");
    add_option("stim_dur_ms", stim_dur_ms_, "Fallback burst duration in ms (used only when f0 is unavailable).");
    add_option("stim_dur_unit", stim_dur_unit_, "Burst duration unit: 'deg' or 'ms' (default: 'deg').");

    add_option("randomize_stim_onset", randomize_stim_onset_, "Whether to randomize stimulus onset phase on each presentation (default: false).");
    add_option("min_stim_dist_sec", min_stim_dist_sec_, "Minimum distance between stimuli in seconds.");
    add_option("max_stim_dist_sec", max_stim_dist_sec_, "Maximum distance between stimuli in seconds.");

    add_option("use_background_sound", use_background_sound_, "Whether to play a continuous background sound (default: false).");
    add_option("background_sound_file", background_sound_file_, "Background sound file name (default: background.wav).");
    add_option("background_dB", background_dB_, "Sound level of signal over background in dB (default: 18).");

    f0_state_ = create_follower_state<double>(
        "f0", 10.0, Permission::NONE,
        "Individual alpha frequency shared by an upstream processor.");
}
void StimControl::CreatePorts()
{
    data_in_port_ = create_input_port<MultiChannelType<double>>(
        "in",
        MultiChannelType<double>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
        PortInPolicy(SlotRange(0, MAX_NCHANNELS)));

    data_out_port_ = create_output_port<MultiChannelType<double>>(
        "out",
        MultiChannelType<double>::Parameters(1, 1, 1), // Placeholder, will be set in CompleteStreamInfo
        PortOutPolicy(SlotRange(0, MAX_NCHANNELS), 200, WaitStrategy::kBlockingStrategy));

    expose_method("set_enabled", &StimControl::SetEnabled);
}

YAML::Node StimControl::SetEnabled(const YAML::Node &node)
{
    bool enabled = node["enabled"].as<bool>();
    stim_enabled_.store(enabled);
    LOG(INFO) << name() << " stimulus output " << (enabled ? "enabled" : "disabled") << " externally.";

    YAML::Node reply;
    reply["enabled"] = enabled;
    return reply;
}

void StimControl::CompleteStreamInfo()
{
    // Set the parameters for the output stream
    for (int k = 0; k < data_out_port_->number_of_slots(); ++k)
    {
        data_out_port_->streaminfo(k).set_stream_rate(data_in_port_->streaminfo(0).stream_rate());
    }
}

bool StimControl::compute_burst_params_(double f0)
{
    double new_burst_ms = 0;

    if (dur_unit_ == DurUnit::kMs)
    {
        // Fixed duration
        new_burst_ms = std::max(1.0, stim_dur_ms_());
    }
    else if (dur_unit_ == DurUnit::kDeg)
    {
        // Duration tracks f0 — burst must be rebuilt whenever f0 changes.
        if (!std::isfinite(f0) || f0 <= 0.0)
        {
            // Fallback to stim_dur_ms_ until a valid f0 arrives.
            new_burst_ms = std::max(1.0, stim_dur_ms_());
        }
        else
        {
            const double deg = std::clamp(stim_dur_deg_(), 0.0, 360.0);
            new_burst_ms = std::clamp((deg / 360.0) / f0, 0.0, 5.0) * 1000.0;   // clamp to 5 seconds max
        }
    }

    // Safety clamp to not exceed the sound buffer length (500ms)
    if (new_burst_ms > burst_precompute_ms_)
    {
        LOG(WARNING) << name() << " Requested burst duration " << new_burst_ms
                     << " ms exceeds burst_precompute_ms_=" << burst_precompute_ms_
                     << " ms; clamping.";
        new_burst_ms = burst_precompute_ms_;
    }

    if (std::abs(new_burst_ms - burst_ms_) > 1) // if burst duration changed by more than 1 ms, update it
    {
        burst_ms_ = new_burst_ms;
        burst_frames_ = std::max(1, (int)(new_burst_ms * fs_audio_ / 1000.0));
        return true;
    }
    else
    {
        return false;
    }
}

void StimControl::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);
    const auto &p = info.parameters<MultiChannelType<double>::Parameters>();
    LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels
              << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate;

    fs_ = p.sample_rate;

    stim_onset_rad_ = stim_onset_deg_() * (1.0 / 180.0 * M_PI);
    LOG(INFO) << name() << " Stimulus onset: " << stim_onset_deg_() << " deg (" << stim_onset_rad_ << " rad)";
    dur_unit_ = (stim_dur_unit_() == "deg") ? DurUnit::kDeg : DurUnit::kMs;

    fs_audio_ = std::max(1, audio_sample_rate_());
    period_ms_ = stim_period_ms_();
    period_frames_ = static_cast<int>(period_ms_ * fs_audio_ / 1000.0);
    LOG(INFO) << name() << " Period frames: " << period_frames_ << " (period_ms=" << period_ms_ << " ms, sample_rate=" << fs_audio_ << " Hz)";

    // Set size to 5 seconds
    background_sound_buffer_ = std::make_unique<moodycamel::ReaderWriterQueue<float>>(5 * fs_audio_ * audio_channels_());

    // Load background sound if enabled
    if (use_background_sound_())
    {
        std::string file = context.resolve_path(background_sound_file_(),"sounds");
        const char *sound_path = file.c_str();
        LOG(INFO) << name() << " Loading background sound from " << sound_path;

        ma_result result;
  
        ma_decoder_config config = ma_decoder_config_init(ma_format_f32, audio_channels_(), fs_audio_);
        result = ma_decoder_init_file(sound_path, &config, &decoder_);
        if (result != MA_SUCCESS)
        {
            LOG(ERROR) << name() << " Failed to initialize decoder for background sound: " << sound_path;
            valid_background_ = false;
        }
        else
        {
            ma_data_source_set_looping(&decoder_, MA_TRUE);
            valid_background_ = true;
        }
    }
}

void StimControl::Preprocess(ProcessingContext &context)
{
    const double f0 = f0_state_ ? f0_state_->get() : std::numeric_limits<double>::quiet_NaN();
    last_f0_ = f0;
    compute_burst_params_(f0); // sets burst_frames_, period_ms_
    build_audio_buffers_();     // uses burst_frames_ and period_ms_ set above

    // Drain any leftover samples from a previous run
    { float tmp; while (background_sound_buffer_->try_dequeue(tmp)) {} }

    if (use_background_sound_() && valid_background_)
    {
        // Fill buffer with 5 seconds of background sound to avoid underruns at start
        int frames = (int)(5 * fs_audio_); // 5 seconds of audio
        std::vector<float> out_f(frames * audio_channels_());
        ma_data_source_read_pcm_frames(&decoder_, out_f.data(), frames, NULL);
        for (float s : out_f)
            background_sound_buffer_->try_enqueue(s);

        // Get power and peak of background sound (first 5 seconds) for gain normalization
        double power_b = 0.0;
        float peak_bg = 0.0f;
        for (float sample : out_f)
        {
            power_b += sample * sample;
            peak_bg = std::max(peak_bg, std::abs(sample));
        }
        power_b /= out_f.size();

        // R = desired amplitude ratio signal/background that achieves the requested dB power ratio.
        // Solving jointly: gain_ * peak_stim + bg_gain_ * peak_bg = 1.0  (no clipping)
        //                  (gain_^2 * power_s_) / (bg_gain_^2 * power_b) = 10^(dB/10)  (dB target)
        // gives: gain_ = R / (R + peak_bg),  bg_gain_ = 1 / (R + peak_bg)
        // where R = sqrt(10^(dB/10) * power_b / power_s_).  peak_stim = 1 (normalised above).
        double R = std::sqrt(std::pow(10.0, background_dB_() / 10.0) * power_b / power_s_);
        gain_    = static_cast<float>(R / (R + peak_bg));
        bg_gain_ = static_cast<float>(1.0 / (R + peak_bg));
        LOG(INFO) << name() << " Signal gain: " << gain_ << ", background gain: " << bg_gain_;

    }
    else {
        // use stimulus amplitude as gain if no background sound
        gain_ = stim_amplitude_();
        bg_gain_ = 0.0f;
        // fill background with zeros
        const int cap = 5 * (int)fs_audio_ * audio_channels_();
        for (int i = 0; i < cap; ++i)
            background_sound_buffer_->try_enqueue(0.0f);
    }

    packet_count_ = 0;
    stimuli_count_ = 0;

    if (!start_audio_())
    {
        LOG(ERROR) << name() << " failed to start audio playback (device: " << audio_device_() << ")";
        throw std::runtime_error("Failed to start audio playback");
    }
}


// Voss-McCartney algorithm (Downey, ThinkDSP)
std::vector<double> voss(int nrows, int ncols, std::mt19937 &rng)
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    std::vector<std::vector<double>> array(
        static_cast<size_t>(nrows),
        std::vector<double>(static_cast<size_t>(ncols), NaN));

    std::uniform_real_distribution<double> unif01(-1.0, 1.0);
    std::uniform_int_distribution<int> row_dist(0, nrows - 1);

    std::geometric_distribution<int> geom(0.5);

    // First row: one random value per column
    for (int c = 0; c < ncols; ++c)
        array[0][c] = unif01(rng);

    // First column: one random value per row
    for (int r = 0; r < nrows; ++r)
        array[r][0] = unif01(rng);

    // n = nrows scattered updates at random (row, col) positions
    const int n = nrows;
    for (int i = 0; i < n; ++i)
    {
        int col = geom(rng) + 1;
        if (col >= ncols)
            col = 0;
        int row = row_dist(rng);
        array[row][col] = unif01(rng); // last write wins on duplicate (row, col)
    }

    // Forward-fill each column: propagate the last valid value downward
    for (int c = 0; c < ncols; ++c)
    {
        double last = array[0][c]; // row 0 is fully populated, always valid
        for (int r = 1; r < nrows; ++r)
        {
            if (std::isnan(array[r][c]))
                array[r][c] = last;
            else
                last = array[r][c];
        }
    }

    // Row-wise sum across all columns
    std::vector<double> total(static_cast<size_t>(nrows), 0.0);
    for (int r = 0; r < nrows; ++r)
    {
        double s = 0.0;
        for (int c = 0; c < ncols; ++c)
            s += array[r][c];
        total[static_cast<size_t>(r)] = s;
    }

    return total;
}

void StimControl::build_audio_buffers_()
{
    const int channels = std::clamp(audio_channels_(), 1, 8);
    const int num_octaves = std::clamp(stim_num_octaves_(), 1, 16);

    // LOG(INFO) << name() << " Building audio buffers: burst=" << burst_frames_
    //           << " frames (" << (static_cast<double>(burst_frames_) / fs_audio_ * 1000.0) << " ms)"
    //           << ", period=" << period_frames_ << " frames";

    int burst_frames_precompute =(int)(burst_precompute_ms_ * fs_audio_ / 1000.0);

    std::mt19937 rng(30); // burst will look the same for each run
    std::vector<double> mono = voss(burst_frames_precompute, num_octaves, rng);

    // Scale to [-1, 1] 
    double peak = 0.0;
    power_s_ = 0.0;
    for (double v : mono)
        peak = std::max(peak, std::abs(v));
    if (peak < 1e-12)
        peak = 1.0;
    for (double &v : mono)
    {
        v = v / peak;
        power_s_ += v * v;
    }
    power_s_ /= mono.size();

    // One-off debug dump for offline inspection (waveform + spectrum) in Python.
    // See extensions/processors/StimControl/plot_burst_buffer.py
    static bool burst_buffer_dumped = false;
    if (!burst_buffer_dumped)
    {
        burst_buffer_dumped = true;
        std::ofstream dbg("results/burst_buffer_debug.txt");
        if (dbg.good())
        {
            dbg << "# fs_audio=" << fs_audio_ << " num_octaves=" << num_octaves << "\n";
            for (double v : mono)
                dbg << v * (-1.0) << "\n";
        }
        else
        {
            LOG(WARNING) << name() << " Could not open results/burst_buffer_debug.txt for burst buffer debug dump";
        }
    }

    sound_buf_.assign(static_cast<size_t>(burst_frames_precompute * channels), 0.0);
    for (int i = 0; i < burst_frames_precompute; ++i)
    {
        for (int ch = 0; ch < channels; ++ch)
        {
            sound_buf_[static_cast<size_t>(i * channels + ch)] = mono[static_cast<size_t>(i)] * (-1);
        }
    }
}

bool StimControl::set_hw_params_interleaved_(snd_pcm_t *pcm,
                                       int sample_rate,
                                       int channels,
                                       snd_pcm_uframes_t period_frames,
                                       snd_pcm_uframes_t buffer_frames,
                                       snd_pcm_format_t format,
                                       const char *format_name,
                                       std::string *fail_step,
                                       int *fail_rc)
{
    snd_pcm_hw_params_t *hw;
    snd_pcm_hw_params_alloca(&hw);

    const auto fail = [&](const char *step, int rc)
    {
        if (fail_step)
            *fail_step = step;
        if (fail_rc)
            *fail_rc = rc;
        return false;
    };

    int rc = 0;
    rc = snd_pcm_hw_params_any(pcm, hw);
    if (rc < 0)
        return fail("snd_pcm_hw_params_any", rc);

    rc = snd_pcm_hw_params_set_access(pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (rc < 0)
        return fail("snd_pcm_hw_params_set_access(RW_INTERLEAVED)", rc);

    rc = snd_pcm_hw_params_test_format(pcm, hw, format);
    if (rc < 0)
    {
        std::string step = std::string("snd_pcm_hw_params_test_format(") + format_name + ")";
        return fail(step.c_str(), rc);
    }

    rc = snd_pcm_hw_params_set_format(pcm, hw, format);
    if (rc < 0)
    {
        std::string step = std::string("snd_pcm_hw_params_set_format(") + format_name + ")";
        return fail(step.c_str(), rc);
    }

    rc = snd_pcm_hw_params_set_channels(pcm, hw, static_cast<unsigned int>(channels));
    if (rc < 0)
        return fail("snd_pcm_hw_params_set_channels", rc);

    unsigned int rate = static_cast<unsigned int>(sample_rate);
    rc = snd_pcm_hw_params_set_rate_near(pcm, hw, &rate, 0);
    if (rc < 0)
        return fail("snd_pcm_hw_params_set_rate_near", rc);

    snd_pcm_uframes_t period = period_frames;
    rc = snd_pcm_hw_params_set_period_size_near(pcm, hw, &period, 0);
    if (rc < 0)
        return fail("snd_pcm_hw_params_set_period_size_near", rc);
    else if (period != period_frames)
    {
        LOG(WARNING) << "ALSA period size changed from requested " << period_frames << " to " << period;
        period_frames_ = period; // update to actual period size
    }

    snd_pcm_uframes_t bufsize = buffer_frames;
    rc = snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &bufsize);
    if (rc < 0)
        return fail("snd_pcm_hw_params_set_buffer_size_near", rc);
    else if (bufsize != buffer_frames)
    {
        LOG(WARNING) << "ALSA buffer size changed from requested " << buffer_frames << " to " << bufsize;
    }

    rc = snd_pcm_hw_params(pcm, hw);
    if (rc < 0)
        return fail("snd_pcm_hw_params(commit)", rc);

    rc = snd_pcm_prepare(pcm);
    if (rc < 0)
        return fail("snd_pcm_prepare", rc);
    return true;
}

static snd_pcm_format_t parse_audio_format_(const std::string &s)
{
    if (s == "s16" || s == "S16" || s == "S16_LE" || s == "s16le" || s == "s16_le")
    {
        return SND_PCM_FORMAT_S16_LE;
    }
    else if (s == "s32" || s == "S32" || s == "S32_LE" || s == "s32le" || s == "s32_le")
    {
        return SND_PCM_FORMAT_S32_LE;
    }
    return SND_PCM_FORMAT_FLOAT_LE;
}

static const char *format_name_(snd_pcm_format_t fmt)
{
    switch (fmt)
    {
    case SND_PCM_FORMAT_S16_LE:
        return "S16_LE";
    case SND_PCM_FORMAT_S32_LE:
        return "S32_LE";
    case SND_PCM_FORMAT_FLOAT_LE:
        return "FLOAT_LE";
    default:
        return "<other>";
    }
}

void StimControl::set_master_volume_(const std::string &card, long pct)
{
    snd_mixer_t *handle;
    if (snd_mixer_open(&handle, 0) < 0)
    {
        LOG(WARNING) << name() << " Could not open ALSA mixer for card " << card;
        return;
    }
    if (snd_mixer_attach(handle, card.c_str()) < 0 ||
        snd_mixer_selem_register(handle, nullptr, nullptr) < 0 ||
        snd_mixer_load(handle) < 0)
    {
        LOG(WARNING) << name() << " Could not load ALSA mixer for card " << card;
        snd_mixer_close(handle);
        return;
    }

    snd_mixer_selem_id_t *sid;
    snd_mixer_selem_id_alloca(&sid);
    snd_mixer_selem_id_set_index(sid, 0);
    snd_mixer_selem_id_set_name(sid, "Master");

    snd_mixer_elem_t *elem = snd_mixer_find_selem(handle, sid);
    if (elem)
    {
        long readback = -1;
        snd_mixer_selem_get_playback_volume(elem, SND_MIXER_SCHN_FRONT_LEFT, &readback);
        LOG(INFO) << name() << " Master volume: current=" << readback << " on card=" << card;

        long min_v, max_v;
        snd_mixer_selem_get_playback_volume_range(elem, &min_v, &max_v);
        long vol = min_v + (max_v - min_v) * pct / 100;
        int rc = snd_mixer_selem_set_playback_volume_all(elem, vol);
        if (rc < 0)
        {
            LOG(WARNING) << name() << " Failed to set Master volume: " << snd_strerror(rc);
        }
        else
        {
            long readback = -1;
            snd_mixer_selem_get_playback_volume(elem, SND_MIXER_SCHN_FRONT_LEFT, &readback);
            LOG(INFO) << name() << " Master volume: target=" << vol << " readback=" << readback
                      << " range=[" << min_v << "," << max_v << "] card=" << card;
        }
    }
    else
    {
        LOG(WARNING) << name() << " 'Master' mixer element not found on " << card;
    }
    snd_mixer_close(handle);
}

bool StimControl::start_audio_()
{
    stop_audio_();

    // Derive card name from device string (e.g. "hw:1,0" -> "hw:1")
    {
        std::string dev = audio_device_();
        auto comma = dev.rfind(',');
        std::string card = (comma != std::string::npos) ? dev.substr(0, comma) : dev;
        set_master_volume_(card, 60);
    }

    const int channels = std::clamp(audio_channels_(), 1, 8);
    const snd_pcm_uframes_t period = static_cast<snd_pcm_uframes_t>(std::max(1, period_frames_));
    // How many frames ALSA can buffer internally; must be >= 2*period_frames_.
    // (older approach below used a fixed 10ms instead of a period-relative size)
    // const snd_pcm_uframes_t bufsize = static_cast<snd_pcm_uframes_t>(std::max(1, fs_audio_ * 10 / 1000));
    const snd_pcm_uframes_t bufsize = static_cast<snd_pcm_uframes_t>(std::max(1, period_frames_ * 3)); // 3 periods

    snd_pcm_t *local_pcm = nullptr;
    const std::string dev = audio_device_();
    int rc = snd_pcm_open(&local_pcm, dev.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    if (rc == -EBUSY)
    {
        int n = sigterm_alsa_device_holders_(dev);
        LOG(WARNING) << name() << " ALSA device '" << dev << "' busy; sent SIGTERM to "
                     << n << " holder(s), retrying...";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        rc = snd_pcm_open(&local_pcm, dev.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    }
    if (rc < 0)
    {
        LOG(ERROR) << name() << " Cannot open ALSA device '" << dev << "': " << snd_strerror(rc);
        return false;
    }

    // Choose format; auto-fallback to S16_LE if FLOAT_LE isn't supported on hw devices.
    snd_pcm_format_t desired_fmt = parse_audio_format_(audio_format_());
    snd_pcm_format_t active_fmt = desired_fmt;

    auto try_config = [&](snd_pcm_format_t fmt, std::string &fail_step, int &fail_rc)
    {
        // ALSA 
        return set_hw_params_interleaved_(local_pcm, fs_audio_, channels, period, bufsize,
                                          fmt, format_name_(fmt), &fail_step, &fail_rc);
    };

    std::string fail_step;
    int fail_rc = 0;
    if (!try_config(desired_fmt, fail_step, fail_rc))
    {
        // Fallback: if user requested float (or defaulted) and the device rejects it, try S16_LE.
        if (desired_fmt == SND_PCM_FORMAT_FLOAT_LE)
        {
            std::string fail_step2;
            int fail_rc2 = 0;
            // Reset hw params state by reopening the device.
            snd_pcm_close(local_pcm);
            local_pcm = nullptr;
            rc = snd_pcm_open(&local_pcm, dev.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
            if (rc < 0)
            {
                LOG(ERROR) << name() << " Cannot reopen ALSA device '" << dev << "' for fallback: " << snd_strerror(rc);
                return false;
            }
            if (try_config(SND_PCM_FORMAT_S16_LE, fail_step2, fail_rc2))
            {
                active_fmt = SND_PCM_FORMAT_S16_LE;
            }
            else
            {
                LOG(ERROR) << name() << " Failed to configure ALSA HW params for '" << dev << "'"
                           << " (step: " << (fail_step.empty() ? "<unknown>" : fail_step) << ", error: "
                           << snd_strerror(fail_rc) << ")"
                           << " [requested rate=" << fs_audio_ << "Hz, channels=" << channels
                           << ", period_frames=" << period << ", buffer_frames=" << bufsize
                           << ", format=" << format_name_(desired_fmt) << "]";
                LOG(ERROR) << name() << " Fallback to S16_LE also failed for '" << dev << "'"
                           << " (step: " << (fail_step2.empty() ? "<unknown>" : fail_step2) << ", error: "
                           << snd_strerror(fail_rc2) << ")";
                snd_pcm_close(local_pcm);
                return false;
            }
        }
        else
        {
            LOG(ERROR) << name() << " Failed to configure ALSA HW params for '" << dev << "'"
                       << " (step: " << (fail_step.empty() ? "<unknown>" : fail_step) << ", error: "
                       << snd_strerror(fail_rc) << ")"
                       << " [requested rate=" << fs_audio_ << "Hz, channels=" << channels
                       << ", period_frames=" << period << ", buffer_frames=" << bufsize
                       << ", format=" << format_name_(desired_fmt) << "]";
            snd_pcm_close(local_pcm);
            return false;
        }
    }

    {
        std::lock_guard<std::mutex> lock(audio_mutex_);
        pcm_ = local_pcm;
        pcm_format_ = active_fmt;
    }
    audio_running_.store(true);
    audio_trigger_pending_.store(false);
    audio_thread_ = std::thread(&StimControl::audio_thread_main_, this);
    return true;
}

void StimControl::stop_audio_() noexcept
{
    audio_running_.store(false);
    audio_trigger_pending_.store(false);

    snd_pcm_t *local_pcm = nullptr;
    {
        std::lock_guard<std::mutex> lock(audio_mutex_);
        local_pcm = pcm_;
    }
    if (local_pcm)
    {
        // Attempt to unblock any blocking write.
        snd_pcm_drop(local_pcm);
    }

    if (audio_thread_.joinable())
    {
        audio_thread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(audio_mutex_);
        if (pcm_)
        {
            snd_pcm_close(pcm_);
            pcm_ = nullptr;
        }
    }
}

snd_pcm_sframes_t StimControl::write_with_recovery_(snd_pcm_t *pcm, const void *buf, snd_pcm_uframes_t frames)
{
    snd_pcm_sframes_t written = snd_pcm_writei(pcm, buf, frames);
    if (written == -EPIPE)
    {
        auto rc = snd_pcm_prepare(pcm);
        if (rc < 0)
            LOG(ERROR) << name() << " Failed to prepare ALSA device after underrun: " << snd_strerror(rc);
        (void)snd_pcm_writei(pcm, buf, frames);
        LOG(WARNING) << name() << " ALSA buffer underrun occurred; attempted recovery";
    }
    else if (written < 0)
    {
        auto rc = snd_pcm_prepare(pcm);
        if (rc < 0)
            LOG(ERROR) << name() << " Failed to prepare ALSA device after underrun: " << snd_strerror(rc);
    }
    else if (written != static_cast<snd_pcm_sframes_t>(frames))
    {
        LOG(WARNING) << name() << " ALSA write returned " << written << " frames, expected " << frames;
    }
    return written;
}

void StimControl::audio_thread_main_()
{
    struct sched_param param;
    param.sched_priority = 80;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0)
        LOG(WARNING) << "Failed to set audio thread RT priority: " << strerror(errno);

    int channels = audio_channels_();
    snd_pcm_sframes_t buffer_size = 1 * fs_audio_ * channels; // 1 second of audio buffer
    std::vector<float>   out_f(buffer_size);
    std::vector<int16_t> out_i16(buffer_size);
    std::vector<int32_t> out_i32(buffer_size);

    snd_pcm_t *local_pcm = nullptr;
    bool do_burst = false;
    snd_pcm_sframes_t frames = 0;
    snd_pcm_sframes_t written;
    // One-pole gain smoother scaffolding: currently disabled (alpha=1.0 means an
    // instant step to target_gain each sample, no ramping). The commented-out
    // expression below is the formula to re-enable a ~kGainRampMs ramp if needed.
    constexpr float kGainRampMs = 2.0f;
    const float gain_ramp_alpha_ = 1.0f; // - std::exp(-1.0f / (kGainRampMs * 0.001f * static_cast<float>(fs_audio_)));

    while (audio_running_.load())
    {
        // TimePoint start_time = Clock::now();
        {
            std::lock_guard<std::mutex> lock(audio_mutex_);
            local_pcm = pcm_;
            if (!local_pcm)
                break;
            // Resolve buffer pointer and frame count under the mutex so we never
            // race with build_audio_buffers_() rewriting these vectors.
            do_burst = audio_trigger_pending_.exchange(false);
            // target_gain = do_burst ? static_cast<float>(stim_amplitude_()) : 0.0f;
            frames = do_burst ? burst_frames_ : period_frames_;
        }
        float target_gain = (do_burst && stim_enabled_.load()) ? gain_ : 0.0f;

        if (frames > 0)
        {
            if (pcm_format_ == SND_PCM_FORMAT_S16_LE)
            {
                for (int i = 0; i < frames * channels; ++i)
                {
                    smoothed_gain_ += (target_gain - smoothed_gain_) * gain_ramp_alpha_;
                    float bg = 0.0f;
                    background_sound_buffer_->try_dequeue(bg);
                    out_i16[i] = float_to_s16_(sound_buf_[i] * smoothed_gain_ + bg * bg_gain_);
                }
                written = write_with_recovery_(local_pcm, out_i16.data(), frames);
            }
            else if (pcm_format_ == SND_PCM_FORMAT_S32_LE)
            {
                for (int i = 0; i < frames * channels; ++i)
                {
                    smoothed_gain_ += (target_gain - smoothed_gain_) * gain_ramp_alpha_;
                    float bg = 0.0f;
                    background_sound_buffer_->try_dequeue(bg);
                    out_i32[i] = float_to_s32(sound_buf_[i] * smoothed_gain_ + bg * bg_gain_);
                }
                written = write_with_recovery_(local_pcm, out_i32.data(), frames);
            }
            else
            {
                for (int i = 0; i < frames * channels; ++i)
                {
                    smoothed_gain_ += (target_gain - smoothed_gain_) * gain_ramp_alpha_;
                    float bg = 0.0f;
                    background_sound_buffer_->try_dequeue(bg);
                    out_f[i] = static_cast<float>(sound_buf_[i]) * smoothed_gain_ + bg * bg_gain_;
                }
                written = write_with_recovery_(local_pcm, out_f.data(), frames);
            }
        }
        // LOG(INFO) << name() << " Audio thread loop time: " << std::chrono::duration<double, std::milli>(Clock::now() - start_time).count() << " ms";
    }

    // Drain only if we still have a valid handle
    {
        std::lock_guard<std::mutex> lock(audio_mutex_);
        local_pcm = pcm_;
    }
    if (local_pcm)
    {
        // drop all queued samples immediately
        snd_pcm_drop(local_pcm);
    }
}

void StimControl::Process(ProcessingContext &context)
{
    MultiChannelType<double>::Data *data_in;
    MultiChannelType<double>::Data *data_out;

    bool valid_f0_and_phase = true;

    // Randomization setup for stimulus timing
    TimePoint last_stim_time_ = Clock::now();
    double time_since_last_stim = 0.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib_interval;
    if (max_stim_dist_sec_() < min_stim_dist_sec_())
    {
        distrib_interval = std::uniform_real_distribution<double>(min_stim_dist_sec_(), min_stim_dist_sec_());
    }
    else
    {
        distrib_interval = std::uniform_real_distribution<double>(min_stim_dist_sec_(), max_stim_dist_sec_());
    }

    double stim_dist_sec_ = distrib_interval(gen);

    LOG(INFO) << name() << " Initial stimulus distance: " << stim_dist_sec_ << " s";

    // Measurement phase
    while (!context.terminated())
    {

        if (n_messages_() != -1 && packet_count_ >= n_messages_())
        {
            break;
        }

        // Try to retrieve all new data
        if (!data_in_port_->slot(0)->RetrieveData(data_in))
        {
            break;
        }
        // At the start of each loop assume that stimulation is valid
        valid_f0_and_phase = true;

        data_out = data_out_port_->slot(0)->ClaimData(false);

        // data_out->CloneTimestamps(*data_in);
        data_out->set_hardware_timestamp(data_in->hardware_timestamp());
        // data_out->set_source_timestamp(data_in->source_timestamp());

        double phase = data_in->data_sample(0, 0);
        if (std::isnan(phase))
        {
            valid_f0_and_phase = false;
            // LOG(WARNING) << name() << " Received NaN phase; skipping stimulation for this packet.";
        }
        const double f0 = f0_state_->get();

        // In "deg" mode the burst duration depends on f0, so burst_ms_/burst_frames_
        // are recomputed here when f0 changes
        if (std::isnan(f0))
        {
            valid_f0_and_phase = false;
            // LOG(WARNING) << name() << " f0 is NaN; skipping stimulation for this packet.";
        }
        else if ((std::isnan(last_f0_) && std::isfinite(f0)) || std::abs(f0 - last_f0_) > 1e-1)
        {
            compute_burst_params_(f0);

            if (!randomize_stim_onset_() && min_stim_dist_sec_() == 0)
            {
                // Allow the next stimuli to fire only at the tail of this cycle
                // Period of a rate 10% faster than f0
                double min_dist = 1 / (f0 + f0 * 0.1); // 10% faster than f0
                distrib_interval = std::uniform_real_distribution<double>(min_dist, min_dist);
                stim_dist_sec_ = distrib_interval(gen);
            }
            last_f0_ = f0;
        }

        TimePoint now = Clock::now();

        uint64_t hw_ts_us = data_in->hardware_timestamp(); // UTC µs of ADC capture
        uint64_t sys_now_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());

        if (sys_now_us < hw_ts_us)
        {
            LOG(WARNING) << name() << " Current time is before hardware timestamp (sys_now: "
                         << sys_now_us << " µs, hw_ts: " << hw_ts_us << " µs).";
        }
        data_in_port_->slot(0)->ReleaseData();

        time_since_last_stim = std::chrono::duration<double>(now - last_stim_time_).count();

        // Still inside a previously triggered stimulus window: output_ keeps
        // mirroring the nominal burst duration independently
        bool in_stim_window = output_samples_remaining_ > 0;

        bool stimulate = false;

        if (!in_stim_window && time_since_last_stim >= stim_dist_sec_)
        {
            if (randomize_stim_onset_())
            {
                stimulate = true;
            }
            else if (valid_f0_and_phase)
            {
                double delay_sec = 0;
                if (correct_latencies_())
                {
                    // System latency
                    delay_sec += static_cast<double>(sys_now_us - hw_ts_us) / 1e6;
                    // Brain latency
                    delay_sec += audio_latency_s_() + erp_latency_s_();
                }

                double phase_advance = 2.0 * M_PI * f0 * delay_sec;
                double diff = phase + phase_advance - stim_onset_rad_;

                // Wrap diff to [0, 2*pi)
                double wrapped_diff = std::fmod(diff, 2.0 * M_PI);
                if (wrapped_diff < 0)
                {
                    wrapped_diff += 2.0 * M_PI;
                }
                // Trigger as soon as the predicted phase enters the first tenths of the
                // target window, so at least 90% of the burst still lands on target
                // even after a bigger phase jump
                stimulate = (wrapped_diff < 0.1 * 2.0 * M_PI); // 10% of an alpha cycle
            }

            if (stimulate) 
            {
                // LOG(INFO) << name() << "Deliver stimulus after: " << time_since_last_stim << " s since last stimulus";
                audio_trigger_pending_.store(true);
                last_stim_time_ = now;
                stimuli_count_++;
                stim_dist_sec_ = distrib_interval(gen);

                // output_ mirrors the full nominal stimulus duration (in samples of
                // the phase stream), independent of how the trigger was decided.
                // Uses burst_ms_ so this stays valid when f0 is NaN, e.g. in fully randomized mode.
                output_samples_remaining_ = std::max(1, (int)std::round(burst_ms_ / 1000.0 * fs_));
                in_stim_window = true;

                // LOG(INFO) << name() << " Packet " << packet_count_ << ": Estimated phase = " << phase << " , delay = " << delay_sec << " s, corr_phase = " << corrected_phase;
            }
        }

        if (in_stim_window)
        {
            output_ = true;
            --output_samples_remaining_;
        }
        else
        {
            output_ = false;
        }

        data_out->set_source_timestamp(Clock::now());
        data_out->set_data_sample(0, 0, static_cast<double>(output_));
        data_out_port_->slot(0)->PublishData();

        if (packet_count_ % static_cast<int>(10 * fs_) == 0)
        {
            LOG(INFO) << name() << " processed " << packet_count_ << " packets, stimuli presented: " << stimuli_count_;
        }
        if (max_n_stimuli_() != -1 && !in_stim_window && stimuli_count_ >= max_n_stimuli_())
        {
            LOG(INFO) << name() << " reached maximum number of stimuli: " << max_n_stimuli_();
            context.Terminate();
        }

        packet_count_++;
        if (packet_count_ % (int)(1*fs_) == 0)
        {
            // Each second, load fill background buffer again until full
            const int total_cap = 5 * (int)fs_audio_ * audio_channels_();
            int buffer_free_size = 0.8 * (total_cap - (int)background_sound_buffer_->size_approx());
            if (use_background_sound_() && valid_background_)
            {
                int frames = buffer_free_size / audio_channels_();
                std::vector<float> out_f(frames * audio_channels_());
                ma_data_source_read_pcm_frames(&decoder_, out_f.data(), frames, NULL);
                for (float s : out_f)
                {
                    bool success = background_sound_buffer_->try_enqueue(s);
                    if (!success)
                    {
                        break;
                    }
                }

            }
            else {
                // fill background with zeros
                for (int i = 0; i < buffer_free_size; ++i)
                {
                    bool success = background_sound_buffer_->try_enqueue(0.0f);
                    if (!success)
                    {
                        break;
                    }
                }
            }


        }
    }
    LOG(INFO) << name() << " stopped working";
}

void StimControl::Postprocess(ProcessingContext &context)
{
    stop_audio_();
    LOG(INFO) << name() << " Total messages processed: " << packet_count_ << ", stimuli presented: " << stimuli_count_;
}

void StimControl::Unprepare(GlobalContext &context)
{
    (void)context; 
    if (use_background_sound_() && valid_background_)
    {
        // ma_device_uninit(&device_);
        ma_decoder_uninit(&decoder_);
    }

    stop_audio_();
}

REGISTERPROCESSOR(StimControl);