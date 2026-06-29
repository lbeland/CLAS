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

#include "StimulusController.hpp"
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
    return static_cast<int16_t>(v * 32767.0f + 0.5f);
}

[[gnu::always_inline]] inline int32_t float_to_s32(float v) {
    return static_cast<int32_t>(v * 8388607.0f) << 8;
}
}

StimulusController::StimulusController() : IProcessor(PRIORITY_HIGH)
{
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("stim_onset_deg", stim_onset_deg_, "Stimulus onset phase in degrees. (<0 -> random)");
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
    add_option("stim_dur_ms", stim_dur_ms_, "Fallback burst duration in ms (used only when IAF is unavailable).");
    add_option("stim_dur_unit", stim_dur_unit_, "Burst duration unit: 'deg' or 'ms' (default: 'deg').");

    add_option("randomize_stim_onset", randomize_stim_onset_, "Whether to randomize stimulus onset phase on each presentation (default: false).");
    add_option("min_stim_dist_sec", min_stim_dist_sec_, "Minimum distance between stimuli in seconds.");
    add_option("max_stim_dist_sec", max_stim_dist_sec_, "Maximum distance between stimuli in seconds.");

    add_option("use_background_sound", use_background_sound_, "Whether to play a continuous background sound (default: false).");
    add_option("background_dB", background_dB_, "Sound level of signal over background in dB (default: 18).");

    iaf_state_ = create_follower_state<double>(
        "iaf", 10.0, Permission::NONE,
        "Individual alpha frequency shared by an upstream processor.");
}
void StimulusController::CreatePorts()
{
    data_in_port_ = create_input_port<MultiChannelType<double>>(
        "in",
        MultiChannelType<double>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
        PortInPolicy(SlotRange(0, MAX_NCHANNELS)));

    data_out_port_ = create_output_port<MultiChannelType<double>>(
        "out",
        MultiChannelType<double>::Parameters(1, 1, 1), // Placeholder, will be set in CompleteStreamInfo
        PortOutPolicy(SlotRange(0, MAX_NCHANNELS), 200, WaitStrategy::kBlockingStrategy));
}

void StimulusController::CompleteStreamInfo()
{
    // Set the parameters for the output stream
    for (int k = 0; k < data_out_port_->number_of_slots(); ++k)
    {
        data_out_port_->streaminfo(k).set_stream_rate(data_in_port_->streaminfo(0).stream_rate());
    }
}

bool StimulusController::compute_burst_params_(double iaf)
{
    double new_burst_ms = 0;

    if (dur_unit_ == DurUnit::kMs)
    {
        // Fixed duration — IAF is irrelevant for burst length.
        // stim_dur_rad_ still uses IAF so the phase window scales with alpha.
        new_burst_ms = std::max(1.0, stim_dur_ms_());
        if (std::isfinite(iaf) && iaf > 0.0)
        {
            stim_dur_rad_ = (new_burst_ms / 1000.0) * (2.0 * M_PI * iaf);
        }
        else
        {
            // No IAF available: assume 10 Hz
            stim_dur_rad_ = (new_burst_ms / 1000.0) * (2.0 * M_PI * 10.0);
            LOG(WARNING) << name() << " IAF unavailable in ms-mode; phase window assumes 10 Hz";
        }
    }
    else if (dur_unit_ == DurUnit::kDeg)
    {
        // Duration tracks IAF — burst must be rebuilt whenever IAF changes.
        if (!std::isfinite(iaf) || iaf <= 0.0)
        {
            // Fallback to stim_dur_ms_ until a valid IAF arrives.
            new_burst_ms = std::max(1.0, stim_dur_ms_());
            stim_dur_rad_ = stim_dur_deg_() * (1.0 / 180.0 * M_PI);
            // LOG(WARNING) << name() << " IAF unavailable in deg-mode; using stim_dur_ms=" << new_burst_ms << " ms as fallback";
        }
        else
        {
            const double deg = std::clamp(stim_dur_deg_(), 0.0, 360.0);
            stim_dur_rad_ = deg * (1.0 / 180.0 * M_PI);
            new_burst_ms = std::clamp((deg / 360.0) / iaf, 0.0, 5.0) * 1000.0;
        }
    }
    else
    {
        // Should never reach here — dur_unit_ is set from validated option in Prepare().
        LOG(ERROR) << name() << " Invalid dur_unit_ enum value";
        throw std::runtime_error("Invalid dur_unit_: must be kMs or kDeg");
    }
    if (std::abs(new_burst_ms - burst_ms_) > 1) // if burst duration changed by more than 1 ms, rebuild buffers
    {
        burst_ms_ = new_burst_ms;
        burst_frames_ = std::max(1, (int)(new_burst_ms * fs_audio_ / 1000.0));
        // period_ms_ = stim_period_ms_(); // silence period
        return true;
    }
    else
    {
        return false;
    }
}

void StimulusController::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);
    const auto &p = info.parameters<MultiChannelType<double>::Parameters>();
    LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels
              << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate;

    fs_ = p.sample_rate;

    if (stim_onset_deg_() < 0)
    {
        LOG(INFO) << name() << " Stimulus onset: random";
    }
    else
    {
        stim_onset_rad_ = stim_onset_deg_() * (1.0 / 180.0 * M_PI);
        LOG(INFO) << name() << " Stimulus onset: " << stim_onset_deg_() << " deg (" << stim_onset_rad_ << " rad)";
    }
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
        std::string file = context.resolve_path("background.mp3","sounds");
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

void StimulusController::Preprocess(ProcessingContext &context)
{
    const double iaf = iaf_state_ ? iaf_state_->get() : std::numeric_limits<double>::quiet_NaN();
    last_iaf_ = iaf;
    compute_burst_params_(iaf); // sets stim_dur_rad_, burst_frames_, period_ms_
    build_audio_buffers_();     // uses burst_frames_ and period_ms_ set above

    // Drain any leftover samples from a previous run
    { float tmp; while (background_sound_buffer_->try_dequeue(tmp)) {} }

    if (use_background_sound_() && valid_background_)
    {
        // Fill buffer with 1 second of background sound to avoid underruns at start
        int frames = (int)(5 * fs_audio_); // 5 seconds of audio
        std::vector<float> out_f(frames * audio_channels_());
        ma_data_source_read_pcm_frames(&decoder_, out_f.data(), frames, NULL);
        for (float s : out_f)
            background_sound_buffer_->try_enqueue(s);

        // Get power and peak of background sound for gain normalization
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

void StimulusController::build_audio_buffers_()
{
    const int channels = std::clamp(audio_channels_(), 1, 8);
    const int num_octaves = std::clamp(stim_num_octaves_(), 1, 16);

    // burst_frames_ and period_ms_ are already set by compute_burst_params_().


    // LOG(INFO) << name() << " Building audio buffers: burst=" << burst_frames_
    //           << " frames (" << (static_cast<double>(burst_frames_) / fs_audio_ * 1000.0) << " ms)"
    //           << ", period=" << period_frames_ << " frames";

    // Pink noise (Voss-McCartney)
    // Seed is fixed for reproducibility: every burst sounds identical, which is
    // intentional for controlled stimulation. Remove the seed for random bursts.
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto white_noise = [&]
    { return dist(rng); };

    std::vector<double> bands(static_cast<size_t>(num_octaves), 0.0);
    unsigned int counter = 0;
    int burst_frames_precompute =(int)(burst_precompute_ms_ * fs_audio_ / 1000.0);

    std::vector<double> mono(static_cast<size_t>(burst_frames_precompute));
    for (int i = 0; i < burst_frames_precompute; ++i)
    {
        ++counter;
        for (int b = 0; b < num_octaves; ++b)
        {
            if ((counter >> b) & 1U)
            {
                bands[static_cast<size_t>(b)] = white_noise();
            }
        }
        double sum = 0.0;
        for (double v : bands)
            sum += v;
        mono[static_cast<size_t>(i)] = sum / static_cast<double>(num_octaves);
    }

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

    sound_buf_.assign(static_cast<size_t>(burst_frames_precompute * channels), 0.0);
    for (int i = 0; i < burst_frames_precompute; ++i)
    {
        for (int ch = 0; ch < channels; ++ch)
        {
            sound_buf_[static_cast<size_t>(i * channels + ch)] = mono[static_cast<size_t>(i)];
        }
    }
}

bool StimulusController::set_hw_params_interleaved_(snd_pcm_t *pcm,
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

bool StimulusController::start_audio_()
{
    stop_audio_();

    const int channels = std::clamp(audio_channels_(), 1, 8);
    const snd_pcm_uframes_t period = static_cast<snd_pcm_uframes_t>(std::max(1, period_frames_));
    // How many frames ALSA can buffer internally; must be >= 2*period_frames_ (10ms)
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
    audio_thread_ = std::thread(&StimulusController::audio_thread_main_, this);
    return true;
}

void StimulusController::stop_audio_() noexcept
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

snd_pcm_sframes_t StimulusController::write_with_recovery_(snd_pcm_t *pcm, const void *buf, snd_pcm_uframes_t frames)
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

void StimulusController::audio_thread_main_()
{
    struct sched_param param;
    param.sched_priority = 95;  // slightly below main (99) but above everything else
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0)
        LOG(WARNING) << "Failed to set audio thread RT priority: " << strerror(errno);

    int channels = audio_channels_();
    snd_pcm_sframes_t buffer_size = 1 * fs_audio_ * channels; // 1 second of audio buffer
    std::vector<float>   out_f(buffer_size);
    std::vector<int16_t> out_i16(buffer_size);
    std::vector<int32_t> out_i32(buffer_size);
    int queue_size_left = 0;
    int queue_size_max = 0;
    int queue_size_min = snd_pcm_avail(pcm_);
    LOG(INFO) << name() << " Audio thread started, initial queue size: " << queue_size_min << " frames";
    snd_pcm_t *local_pcm = nullptr;
    bool do_burst = false;
    snd_pcm_sframes_t frames = 0;
    snd_pcm_sframes_t written = 0;

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
        float target_gain = do_burst ? gain_ : 0.0f;

        if (frames > 0)
        {
            if (pcm_format_ == SND_PCM_FORMAT_S16_LE)
            {
                for (int i = 0; i < frames * channels; ++i)
                {
                    // gain_ is directly set to target, change 1 to a smaller value for smoothing
                    // gain_ += (target_gain - gain_) * 0.01; // 0.01f;
                    float bg = 0.0f;
                    background_sound_buffer_->try_dequeue(bg);
                    out_i16[i] = float_to_s16_(sound_buf_[i] * target_gain + bg * bg_gain_);
                }
                written = write_with_recovery_(local_pcm, out_i16.data(), frames);
            }
            else if (pcm_format_ == SND_PCM_FORMAT_S32_LE)
            {
                for (int i = 0; i < frames * channels; ++i)
                {
                    // gain_ += (target_gain - gain_) * 0.01; // 0.01f;
                    float bg = 0.0f;
                    background_sound_buffer_->try_dequeue(bg);
                    out_i32[i] = float_to_s32(sound_buf_[i] * target_gain + bg * bg_gain_);
                }
                written = write_with_recovery_(local_pcm, out_i32.data(), frames);
            }
            else
            {
                for (int i = 0; i < frames * channels; ++i)
                {
                    // gain_ += (target_gain - gain_) * 0.01; // 0.01f;
                    float bg = 0.0f;
                    background_sound_buffer_->try_dequeue(bg);
                    out_f[i] = static_cast<float>(sound_buf_[i]) * target_gain + bg * bg_gain_;
                }
                written = write_with_recovery_(local_pcm, out_f.data(), frames);
            }
        }
        // LOG(INFO) << name() << " Audio thread loop time: " << std::chrono::duration<double, std::milli>(Clock::now() - start_time).count() << " ms";
    }

    LOG(INFO) << name() << "Max queue size:: " << queue_size_max << " frames, Min queue size: " << queue_size_min << " frames";
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

void StimulusController::Process(ProcessingContext &context)
{

    MultiChannelType<double>::Data *data_in;
    MultiChannelType<double>::Data *data_out;

    bool valid_stimulation = true;

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

    std::uniform_real_distribution<double> distrib_onset(0.0, 2.0 * M_PI);
    double stim_dist_sec_ = distrib_interval(gen);
    if (randomize_stim_onset_())
    {
        stim_onset_rad_ = distrib_onset(gen);
    }
    LOG(INFO) << name() << " Initial stimulus distance: " << stim_dist_sec_ << " s, onset: " << stim_onset_rad_ << " rad";

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
        valid_stimulation = true;

        data_out = data_out_port_->slot(0)->ClaimData(false);

        // data_out->CloneTimestamps(*data_in);
        data_out->set_hardware_timestamp(data_in->hardware_timestamp());
        // data_out->set_source_timestamp(data_in->source_timestamp());

        double phase = data_in->data_sample(0, 0);
        if (std::isnan(phase))
        {
            valid_stimulation = false;
            // LOG(WARNING) << name() << " Received NaN phase; skipping stimulation for this packet.";
        }
        const double iaf_ = iaf_state_->get();

        // In "deg" mode the burst duration depends on IAF. Rebuild buffers
        // when IAF changes (guarded by audio_mutex_ so the audio thread is safe).
        if (std::isnan(iaf_))
        {
            valid_stimulation = false;
            // LOG(WARNING) << name() << " IAF is NaN; skipping stimulation for this packet.";
        }
        else if ((std::isnan(last_iaf_) && std::isfinite(iaf_)) || std::abs(iaf_ - last_iaf_) > 1e-1)
        {
            const bool frames_changed = compute_burst_params_(iaf_);
            // if (frames_changed)
            // {
            //     std::lock_guard<std::mutex> lock(audio_mutex_);
            //     build_audio_buffers_();
            //     LOG(INFO) << name() << " IAF changed " << last_iaf_ << " -> " << iaf_
            //               << ": burst rebuilt to " << burst_ms_ << " ms";
            // }
            if (min_stim_dist_sec_() == 0)
            {
                // make sure each stimulus is minimum half an alpha cycle apart to avoid overlapping bursts
                // double half_alpha_cycle = 1.0 / (2.0 * iaf_);
                double min_dist = 1 / (iaf_ + iaf_ * 0.1); // 10% faster than IAF
                distrib_interval = std::uniform_real_distribution<double>(min_dist, min_dist);
                stim_dist_sec_ = distrib_interval(gen);
            }
            last_iaf_ = iaf_;
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

        if (valid_stimulation && time_since_last_stim >= stim_dist_sec_ || output_)
        {
            double delay_sec = 0;
            if (correct_latencies_())
            {
                // System latency
                delay_sec += static_cast<double>(sys_now_us - hw_ts_us) / 1e6;
                // Brain latency
                delay_sec += audio_latency_s_() + erp_latency_s_();
            }

            double phase_advance = 2.0 * M_PI * iaf_ * delay_sec;
            double diff = phase + phase_advance - stim_onset_rad_;

            // Wrap diff to [-pi, pi] once
            const double wrapped_diff = std::atan2(std::sin(diff), std::cos(diff));

            output_ = std::abs(wrapped_diff) < (stim_dur_rad_ / 2);

            // Trigger a single burst on the rising edge (false -> true)
            if (output_ && !last_output_)
            {
                // LOG(INFO) << name() << "Deliver stimulus after: " << time_since_last_stim << " s since last stimulus";
                audio_trigger_pending_.store(true);
                last_stim_time_ = now;
                stimuli_count_++;
                stim_dist_sec_ = distrib_interval(gen);
                if (randomize_stim_onset_())
                {
                    stim_onset_rad_ = distrib_onset(gen);
                }

                // LOG(INFO) << name() << " Packet " << packet_count_ << ": Estimated phase = " << phase << " , delay = " << delay_sec << " s, corr_phase = " << corrected_phase;
            }
        }
        else
        {
            output_ = false;
        }

        last_output_ = output_;

        data_out->set_source_timestamp( Clock::now());
        data_out->set_data_sample(0, 0, static_cast<double>(output_));
        data_out_port_->slot(0)->PublishData();

        packet_count_++;
        if (packet_count_ % (int)fs_ == 0)
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

void StimulusController::Postprocess(ProcessingContext &context)
{
    stop_audio_();
    LOG(INFO) << name() << " Total messages processed: " << packet_count_ << ", stimuli presented: " << stimuli_count_;
}

void StimulusController::Unprepare(GlobalContext &context)
{
    (void)context; 
    if (use_background_sound_() && valid_background_)
    {
        // ma_device_uninit(&device_);
        ma_decoder_uninit(&decoder_);
    }

    stop_audio_();
}

REGISTERPROCESSOR(StimulusController);