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

static inline int16_t double_to_s16_(double x)
{
    if (x > 1.0)
        x = 1.0;
    if (x < -1.0)
        x = -1.0;
    // symmetric mapping; -1.0 maps to -32767 to avoid overflow on int16
    return static_cast<int16_t>(lrint(x * 32767.0));
}

double wrap_phase_rad(double radians)
{
    return std::fmod(radians + M_PI, 2.0 * M_PI) - M_PI;
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
    int new_burst_frames = 0;

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
}

void StimulusController::Preprocess(ProcessingContext &context)
{
    const double iaf = iaf_state_ ? iaf_state_->get() : std::numeric_limits<double>::quiet_NaN();
    last_iaf_ = iaf;
    compute_burst_params_(iaf); // sets stim_dur_rad_, burst_frames_, period_ms_
    build_audio_buffers_();     // uses burst_frames_ and period_ms_ set above

    if (!start_audio_())
    {
        LOG(ERROR) << name() << " failed to start audio playback (device: " << audio_device_() << ")";
        throw std::runtime_error("Failed to start audio playback");
    }

    packet_count_ = 0;
    stimuli_count_ = 0;
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

    std::vector<double> mono(static_cast<size_t>(burst_frames_));
    for (int i = 0; i < burst_frames_; ++i)
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
    for (double v : mono)
        peak = std::max(peak, std::abs(v));
    if (peak < 1e-12)
        peak = 1.0;
    for (double &v : mono)
        v = v / peak;

    sound_buf_.assign(static_cast<size_t>(burst_frames_ * channels), 0.0);
    for (int i = 0; i < burst_frames_; ++i)
    {
        for (int ch = 0; ch < channels; ++ch)
        {
            sound_buf_[static_cast<size_t>(i * channels + ch)] = mono[static_cast<size_t>(i)];
        }
    }

}

static bool set_hw_params_interleaved_(snd_pcm_t *pcm,
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

    snd_pcm_uframes_t bufsize = buffer_frames;
    rc = snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &bufsize);
    if (rc < 0)
        return fail("snd_pcm_hw_params_set_buffer_size_near", rc);

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
    return SND_PCM_FORMAT_FLOAT_LE;
}

static const char *format_name_(snd_pcm_format_t fmt)
{
    switch (fmt)
    {
    case SND_PCM_FORMAT_S16_LE:
        return "S16_LE";
    case SND_PCM_FORMAT_FLOAT_LE:
        return "FLOAT_LE";
    default:
        return "<other>";
    }
}

bool StimulusController::start_audio_()
{
    stop_audio_();

    const int sample_rate = std::max(1, audio_sample_rate_());
    const int channels = std::clamp(audio_channels_(), 1, 8);
    const snd_pcm_uframes_t period = static_cast<snd_pcm_uframes_t>(std::max(1, period_frames_));
    const snd_pcm_uframes_t bufsize = period * 2; // How many frames ALSA should buffer internally; must be >= period_frames_

    snd_pcm_t *local_pcm = nullptr;
    const std::string dev = audio_device_();
    int rc = snd_pcm_open(&local_pcm, dev.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
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

void StimulusController::write_with_recovery_(snd_pcm_t *pcm, const void *buf, snd_pcm_uframes_t frames)
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
}

void StimulusController::audio_thread_main_()
{
    while (audio_running_.load())
    {
        snd_pcm_t *local_pcm = nullptr;
        bool do_burst = false;
        float target_gain = 0.0f;
        snd_pcm_uframes_t frames = 0;

        {
            std::lock_guard<std::mutex> lock(audio_mutex_);
            local_pcm = pcm_;
            if (!local_pcm) break;
            // Resolve buffer pointer and frame count under the mutex so we never
            // race with build_audio_buffers_() rewriting these vectors.
            do_burst = audio_trigger_pending_.exchange(false);
            target_gain = do_burst ? static_cast<float>(stim_amplitude_()) : 0.0f;

            // Determine frame count
            frames = do_burst ? burst_frames_ : period_frames_;
        }


        snd_pcm_sframes_t written;

        // Apply gain and convert inline — no second buffer needed
        if (pcm_format_ == SND_PCM_FORMAT_S16_LE)
            {
            std::vector<int16_t> out(frames * audio_channels_());
            for (int i = 0; i < frames * audio_channels_(); ++i)
            {
                // gain_ is directly set to target, change 1 to a smaller value for smoothing
                gain_ += (target_gain - gain_) * 0.01; //0.01f;
                out[i] = double_to_s16_(sound_buf_[i] * gain_);
            }
            write_with_recovery_(local_pcm, out.data(), frames);
            
        }
        else
        {
            std::vector<float> out(frames * audio_channels_());
            for (int i = 0; i < frames * audio_channels_(); ++i)
        {
                gain_ += (target_gain - gain_) * 0.01; //0.01f;
                out[i] = static_cast<float>(sound_buf_[i]) * gain_;
            }
            write_with_recovery_(local_pcm, out.data(), frames);
        }
        
    }

    // Drain only if we still have a valid handle
    snd_pcm_t *local_pcm = nullptr;
    {
        std::lock_guard<std::mutex> lock(audio_mutex_);
        local_pcm = pcm_;
    }
    if (local_pcm)
    {
        // drop all queued sampels immediately
        snd_pcm_drop(local_pcm);
    }
}

void StimulusController::Process(ProcessingContext &context)
{
    MultiChannelType<double>::Data *data_in;
    MultiChannelType<double>::Data *data_out;

    bool valid_stimulation = true;

    // Randomization setup for stimulus timing (if randomize_stim_onset_() is enabled)
    TimePoint last_stim_time_ = Clock::now();
    double time_since_last_stim = 0.0;
    std::random_device rd;  // non-deterministic generator
    std::mt19937 gen(rd()); // to seed mersenne twister.
                            // replace the call to rd() with a
                            // constant value to get repeatable
                            // results.
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
            if (frames_changed)
            {
                std::lock_guard<std::mutex> lock(audio_mutex_);
                build_audio_buffers_();
                LOG(INFO) << name() << " IAF changed " << last_iaf_ << " -> " << iaf_
                          << ": burst rebuilt to " << burst_ms_ << " ms";
            }
            if (min_stim_dist_sec_() == 0){
                // make sure each stimulus is minimum half an alpha cycle apart to avoid overlapping bursts 
                double half_alpha_cycle = 1.0 / (2.0 * iaf_);   
                distrib_interval = std::uniform_real_distribution<double>(half_alpha_cycle, half_alpha_cycle);
                stim_dist_sec_ = distrib_interval(gen);
            }
            last_iaf_ = iaf_;
        }

        TimePoint now = Clock::now();

        TimePoint sample_ts = data_in->source_timestamp();
        if (now < sample_ts)
        {
            LOG(WARNING) << name() << " Current time is before sample timestamp (now: " << std::chrono::duration<double, std::milli>(now.time_since_epoch()).count()
                         << " ms, sample_ts: " << std::chrono::duration<double, std::milli>(sample_ts.time_since_epoch()).count() << " ms).";
        }
        // data_out->set_source_timestamp(now);
        data_in_port_->slot(0)->ReleaseData();

        time_since_last_stim = std::chrono::duration<double>(now - last_stim_time_).count();

        if (valid_stimulation && time_since_last_stim >= stim_dist_sec_)
        {
            double delay_sec = 0;
            if (correct_latencies_())
            {
                // System latency
                delay_sec += std::chrono::duration<double>(now - sample_ts).count();
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

                // LOG(INFO) << name() << " Packet " << packet_count_ << ": Estimated phase = " << phase << " , delay = " << delay_sec << " s, corr_phase = " << corrected_phase;
            }
            else if (!output_ && last_output_)
            {
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

        TimePoint timestamp = Clock::now();
        last_output_ = output_;

        data_out->set_source_timestamp(timestamp);
        data_out->set_data_sample(0, 0, static_cast<double>(output_));
        data_out_port_->slot(0)->PublishData();

        packet_count_++;
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
    stop_audio_();
}

REGISTERPROCESSOR(StimulusController);