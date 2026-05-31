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

static inline int16_t double_to_s16_(double x) {
    if (x > 1.0) x = 1.0;
    if (x < -1.0) x = -1.0;
    // symmetric mapping; -1.0 maps to -32767 to avoid overflow on int16
    return static_cast<int16_t>(lrint(x * 32767.0));
}

StimulusController::StimulusController() : IProcessor(PRIORITY_HIGH)
{
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("stim_onset_deg", stim_onset_deg_, "Stimulus onset phase in degrees.");
    add_option("audio_latency_s", audio_latency_s_, "Estimated audio latency in seconds (for phase correction).");
    add_option("erp_latency_s", erp_latency_s_, "Estimated auditory evoked response potential latency in seconds (for phase correction).");

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

bool StimulusController::compute_burst_params_(double iaf) {
    const int sample_rate = std::max(1, audio_sample_rate_());
    int new_burst_frames = 0;

    if (dur_unit_ == DurUnit::kMs) {
        // Fixed duration — IAF is irrelevant for burst length.
        // stim_dur_rad_ still uses IAF so the phase window scales with alpha.
        const int burst_ms = std::max(1, stim_dur_ms_());
        new_burst_frames = burst_ms * sample_rate / 1000;
        if (std::isfinite(iaf) && iaf > 0.0) {
            stim_dur_rad_ = (burst_ms / 1000.0) * (2.0 * M_PI * iaf);
        } else {
            // No IAF available: assume 10 Hz
            stim_dur_rad_ = (burst_ms / 1000.0) * (2.0 * M_PI * 10.0);
            LOG(WARNING) << name() << " IAF unavailable in ms-mode; phase window assumes 10 Hz";
        }

    } else if (dur_unit_ == DurUnit::kDeg) {
        // Duration tracks IAF — burst must be rebuilt whenever IAF changes.
        if (!std::isfinite(iaf) || iaf <= 0.0) {
            // Fallback to stim_dur_ms_ until a valid IAF arrives.
            const int burst_ms = std::max(1, stim_dur_ms_());
            new_burst_frames = burst_ms * sample_rate / 1000;
            stim_dur_rad_ = stim_dur_deg_() * (1.0 / 180.0 * M_PI);
            // LOG(WARNING) << name() << " IAF unavailable in deg-mode; using stim_dur_ms=" << burst_ms << " ms as fallback";
        } else {
            const double deg = std::clamp(stim_dur_deg_(), 0.0, 360.0);
            const double duration_sec = std::clamp((deg / 360.0) / iaf, 0.0, 5.0);
            new_burst_frames = static_cast<int>(std::lround(duration_sec * sample_rate));
            stim_dur_rad_ = deg * (1.0 / 180.0 * M_PI);
        }

    } else {
        // Should never reach here — dur_unit_ is set from validated option in Prepare().
        LOG(ERROR) << name() << " Invalid dur_unit_ enum value";
        throw std::runtime_error("Invalid dur_unit_: must be kMs or kDeg");
    }
    LOG(INFO) << name() << "Stimulus duration: " << stim_dur_rad_ << " rad, " << new_burst_frames << " frames at " << sample_rate << " Hz (IAF: " << iaf << " Hz)";
    new_burst_frames = std::max(1, new_burst_frames);
    const bool changed = (new_burst_frames != burst_frames_);
    burst_frames_ = new_burst_frames;
    period_ms_ = stim_period_ms_(); // silence period
    return changed;
}

void StimulusController::Prepare(GlobalContext &context)
{
    const auto &info = data_in_port_->streaminfo(0);
    const auto &p = info.parameters<MultiChannelType<double>::Parameters>();
    LOG(INFO) << name() << " Input Stream parameters - nchannels: " << p.nchannels
              << ", nsamples: " << p.nsamples << ", sample_rate: " << p.sample_rate;
    
    fs_ = p.sample_rate;

    stim_onset_rad_ = stim_onset_deg_() * (1.0 / 180.0 * M_PI);
    LOG(INFO) << name() << " Stimulus onset: " << stim_onset_deg_() << " deg (" << stim_onset_rad_ << " rad)";
    dur_unit_ = (stim_dur_unit_() == "deg") ? DurUnit::kDeg : DurUnit::kMs;

    const double iaf = iaf_state_ ? iaf_state_->get() : std::numeric_limits<double>::quiet_NaN();
    last_iaf_ = iaf;
    compute_burst_params_(iaf);   // sets stim_dur_rad_, burst_frames_, period_ms_

    build_audio_buffers_();       // uses burst_frames_ and period_ms_ set above

    if (!start_audio_()) {
        LOG(ERROR) << name() << " failed to start audio playback (device: " << audio_device_() << ")";
        throw std::runtime_error("Failed to start audio playback");
    }
}

void StimulusController::build_audio_buffers_() {
    const int sample_rate = std::max(1, audio_sample_rate_());
    const int channels = std::clamp(audio_channels_(), 1, 8);
    const double amplitude = std::clamp(stim_amplitude_(), 0.0, 1.0);
    const int num_octaves = std::clamp(stim_num_octaves_(), 1, 16);

    // burst_frames_ and period_ms_ are already set by compute_burst_params_().
    period_frames_ = static_cast<int>(period_ms_ * sample_rate / 1000.0);
    if (burst_frames_ <= 0) burst_frames_ = 1;
    if (period_frames_ <= 0) period_frames_ = 1;

    LOG(INFO) << name() << " Building audio buffers: burst=" << burst_frames_
              << " frames (" << (static_cast<double>(burst_frames_) / sample_rate * 1000.0) << " ms)"
              << ", period=" << period_frames_ << " frames";

    // Pink noise (Voss-McCartney)
    // Seed is fixed for reproducibility: every burst sounds identical, which is
    // intentional for controlled stimulation. Remove the seed for random bursts.
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto white_noise = [&] { return dist(rng); };

    std::vector<double> bands(static_cast<size_t>(num_octaves), 0.0);
    unsigned int counter = 0;

    std::vector<double> mono(static_cast<size_t>(burst_frames_));
    for (int i = 0; i < burst_frames_; ++i) {
        ++counter;
        for (int b = 0; b < num_octaves; ++b) {
            if ((counter >> b) & 1U) {
                bands[static_cast<size_t>(b)] = white_noise();
            }
        }
        double sum = 0.0;
        for (double v : bands) sum += v;
        mono[static_cast<size_t>(i)] = sum / static_cast<double>(num_octaves);
    }

    double peak = 0.0;
    for (double v : mono) peak = std::max(peak, std::abs(v));
    if (peak < 1e-12) peak = 1.0;
    for (double &v : mono) v = (v / peak) * amplitude;

    burst_buf_.assign(static_cast<size_t>(burst_frames_ * channels), 0.0);
    for (int i = 0; i < burst_frames_; ++i) {
        for (int ch = 0; ch < channels; ++ch) {
            burst_buf_[static_cast<size_t>(i * channels + ch)] = mono[static_cast<size_t>(i)];
        }
    }

    silence_buf_.assign(static_cast<size_t>(period_frames_ * channels), 0.0);

    burst_buf_s16_.assign(burst_buf_.size(), 0);
    for (size_t i = 0; i < burst_buf_.size(); ++i) {
        burst_buf_s16_[i] = double_to_s16_(burst_buf_[i]);
    }
    silence_buf_s16_.assign(silence_buf_.size(), 0);
}

static bool set_hw_params_interleaved_(snd_pcm_t* pcm,
                                             int sample_rate,
                                             int channels,
                                             snd_pcm_uframes_t period_frames,
                                             snd_pcm_uframes_t buffer_frames,
                                             snd_pcm_format_t format,
                                             const char* format_name,
                                             std::string* fail_step,
                                             int* fail_rc) {
    snd_pcm_hw_params_t* hw;
    snd_pcm_hw_params_alloca(&hw);

    const auto fail = [&](const char* step, int rc) {
        if (fail_step) *fail_step = step;
        if (fail_rc) *fail_rc = rc;
        return false;
    };

    int rc = 0;
    rc = snd_pcm_hw_params_any(pcm, hw);
    if (rc < 0) return fail("snd_pcm_hw_params_any", rc);

    rc = snd_pcm_hw_params_set_access(pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (rc < 0) return fail("snd_pcm_hw_params_set_access(RW_INTERLEAVED)", rc);

    rc = snd_pcm_hw_params_test_format(pcm, hw, format);
    if (rc < 0) {
        std::string step = std::string("snd_pcm_hw_params_test_format(") + format_name + ")";
        return fail(step.c_str(), rc);
    }

    rc = snd_pcm_hw_params_set_format(pcm, hw, format);
    if (rc < 0) {
        std::string step = std::string("snd_pcm_hw_params_set_format(") + format_name + ")";
        return fail(step.c_str(), rc);
    }

    rc = snd_pcm_hw_params_set_channels(pcm, hw, static_cast<unsigned int>(channels));
    if (rc < 0) return fail("snd_pcm_hw_params_set_channels", rc);

    unsigned int rate = static_cast<unsigned int>(sample_rate);
    rc = snd_pcm_hw_params_set_rate_near(pcm, hw, &rate, 0);
    if (rc < 0) return fail("snd_pcm_hw_params_set_rate_near", rc);

    snd_pcm_uframes_t period = period_frames;
    rc = snd_pcm_hw_params_set_period_size_near(pcm, hw, &period, 0);
    if (rc < 0) return fail("snd_pcm_hw_params_set_period_size_near", rc);

    snd_pcm_uframes_t bufsize = buffer_frames;
    rc = snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &bufsize);
    if (rc < 0) return fail("snd_pcm_hw_params_set_buffer_size_near", rc);

    rc = snd_pcm_hw_params(pcm, hw);
    if (rc < 0) return fail("snd_pcm_hw_params(commit)", rc);

    rc = snd_pcm_prepare(pcm);
    if (rc < 0) return fail("snd_pcm_prepare", rc);
    return true;
}

static snd_pcm_format_t parse_audio_format_(const std::string& s) {
    if (s == "s16" || s == "S16" || s == "S16_LE" || s == "s16le" || s == "s16_le") {
        return SND_PCM_FORMAT_S16_LE;
    }
    return SND_PCM_FORMAT_FLOAT_LE;
}

static const char* format_name_(snd_pcm_format_t fmt) {
    switch (fmt) {
        case SND_PCM_FORMAT_S16_LE: return "S16_LE";
        case SND_PCM_FORMAT_FLOAT_LE: return "FLOAT_LE";
        default: return "<other>";
    }
}

bool StimulusController::start_audio_() {
    stop_audio_();

    const int sample_rate = std::max(1, audio_sample_rate_());
    const int channels = std::clamp(audio_channels_(), 1, 8);
    const snd_pcm_uframes_t period = static_cast<snd_pcm_uframes_t>(std::max(1, period_frames_));
    const snd_pcm_uframes_t bufsize = period * 2;   // How many frames ALSA should buffer internally; must be >= period_frames_

    snd_pcm_t* local_pcm = nullptr;
    const std::string dev = audio_device_();
    int rc = snd_pcm_open(&local_pcm, dev.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    if (rc < 0) {
        LOG(ERROR) << name() << " Cannot open ALSA device '" << dev << "': " << snd_strerror(rc);
        return false;
    }

    // Choose format; auto-fallback to S16_LE if FLOAT_LE isn't supported on hw devices.
    snd_pcm_format_t desired_fmt = parse_audio_format_(audio_format_());
    snd_pcm_format_t active_fmt = desired_fmt;

    auto try_config = [&](snd_pcm_format_t fmt, std::string& fail_step, int& fail_rc) {
        return set_hw_params_interleaved_(local_pcm, sample_rate, channels, period, bufsize,
                                          fmt, format_name_(fmt), &fail_step, &fail_rc);
    };

    std::string fail_step;
    int fail_rc = 0;
    if (!try_config(desired_fmt, fail_step, fail_rc)) {
        // Fallback: if user requested float (or defaulted) and the device rejects it, try S16_LE.
        if (desired_fmt == SND_PCM_FORMAT_FLOAT_LE) {
            std::string fail_step2;
            int fail_rc2 = 0;
            // Reset hw params state by reopening the device.
            snd_pcm_close(local_pcm);
            local_pcm = nullptr;
            rc = snd_pcm_open(&local_pcm, dev.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
            if (rc < 0) {
                LOG(ERROR) << name() << " Cannot reopen ALSA device '" << dev << "' for fallback: " << snd_strerror(rc);
                return false;
            }
            if (try_config(SND_PCM_FORMAT_S16_LE, fail_step2, fail_rc2)) {
                active_fmt = SND_PCM_FORMAT_S16_LE;
            } else {
                LOG(ERROR) << name() << " Failed to configure ALSA HW params for '" << dev << "'"
                           << " (step: " << (fail_step.empty() ? "<unknown>" : fail_step) << ", error: "
                           << snd_strerror(fail_rc) << ")"
                           << " [requested rate=" << sample_rate << "Hz, channels=" << channels
                           << ", period_frames=" << period << ", buffer_frames=" << bufsize
                           << ", format=" << format_name_(desired_fmt) << "]";
                LOG(ERROR) << name() << " Fallback to S16_LE also failed for '" << dev << "'"
                           << " (step: " << (fail_step2.empty() ? "<unknown>" : fail_step2) << ", error: "
                           << snd_strerror(fail_rc2) << ")";
                snd_pcm_close(local_pcm);
                return false;
            }
        } else {
            LOG(ERROR) << name() << " Failed to configure ALSA HW params for '" << dev << "'"
                       << " (step: " << (fail_step.empty() ? "<unknown>" : fail_step) << ", error: "
                       << snd_strerror(fail_rc) << ")"
                       << " [requested rate=" << sample_rate << "Hz, channels=" << channels
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

void StimulusController::stop_audio_() noexcept {
    audio_running_.store(false);
    audio_trigger_pending_.store(false);

    snd_pcm_t* local_pcm = nullptr;
    {
        std::lock_guard<std::mutex> lock(audio_mutex_);
        local_pcm = pcm_;
    }
    if (local_pcm) {
        // Attempt to unblock any blocking write.
        snd_pcm_drop(local_pcm);
    }

    if (audio_thread_.joinable()) {
        audio_thread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(audio_mutex_);
        if (pcm_) {
            snd_pcm_close(pcm_);
            pcm_ = nullptr;
        }
    }
}

void StimulusController::audio_thread_main_() {
    while (audio_running_.load()) {
        snd_pcm_t* local_pcm = nullptr;
        const void* buf = nullptr;
        snd_pcm_uframes_t frames = 0;

        {
            std::lock_guard<std::mutex> lock(audio_mutex_);
            local_pcm = pcm_;
            if (!local_pcm) break;

            // Resolve buffer pointer and frame count under the mutex so we never
            // race with build_audio_buffers_() rewriting these vectors.
            const bool do_burst = audio_trigger_pending_.exchange(false);
            frames = static_cast<snd_pcm_uframes_t>(do_burst ? burst_frames_ : period_frames_);
            if (pcm_format_ == SND_PCM_FORMAT_S16_LE) {
                buf = do_burst ? static_cast<const void*>(burst_buf_s16_.data())
                               : static_cast<const void*>(silence_buf_s16_.data());
            } else {
                buf = do_burst ? static_cast<const void*>(burst_buf_.data())
                               : static_cast<const void*>(silence_buf_.data());
            }
        }
        // buf points into vector heap storage. Reallocation only happens inside
        // build_audio_buffers_() which requires audio_mutex_, so the pointer is
        // stable for the duration of this (blocking) write.
        snd_pcm_sframes_t written = snd_pcm_writei(local_pcm, buf, frames);
        if (written == -EPIPE) {
            auto rc = snd_pcm_prepare(local_pcm);
            if (rc < 0) {
                LOG(ERROR) << name() << " Failed to prepare ALSA device after underrun: " << snd_strerror(rc);
            }
            // Retry once after xrun recovery
            (void)snd_pcm_writei(local_pcm, buf, frames);
            LOG(WARNING) << name() << " ALSA buffer underrun occurred; attempted recovery";
        } else if (written < 0) {
            // Other recoverable errors (e.g. suspended)
            auto rc = snd_pcm_prepare(local_pcm);
            if (rc < 0) {
                LOG(ERROR) << name() << " Failed to prepare ALSA device after underrun: " << snd_strerror(rc);
            }
        }
        // Short-write: remaining frames will be covered by the next silence period.
    }

    // Drain only if we still have a valid handle
    snd_pcm_t* local_pcm = nullptr;
    {
        std::lock_guard<std::mutex> lock(audio_mutex_);
        local_pcm = pcm_;
    }
    if (local_pcm) {
        snd_pcm_drain(local_pcm);
    }
}


void StimulusController::Process(ProcessingContext &context)
{
    MultiChannelType<double>::Data *data_in;
    MultiChannelType<double>::Data *data_out;

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

        data_out = data_out_port_->slot(0)->ClaimData(false);
        // data_out->CloneTimestamps(*data_in);
        data_out->set_hardware_timestamp(data_in->hardware_timestamp());
        data_out->set_source_timestamp(data_in->source_timestamp());
        double phase = data_in->data_sample(0, 0);
        const double iaf_ = iaf_state_->get();

        // In "deg" mode the burst duration depends on IAF. Rebuild buffers
        // when IAF changes (guarded by audio_mutex_ so the audio thread is safe).
        if ((std::isnan(last_iaf_) && std::isfinite(iaf_)) || iaf_ - last_iaf_ > 1e-2) {
            const bool frames_changed = compute_burst_params_(iaf_);
            if (frames_changed) {
                std::lock_guard<std::mutex> lock(audio_mutex_);
                build_audio_buffers_();
                LOG(INFO) << name() << " IAF changed " << last_iaf_ << " -> " << iaf_
                        << ": burst rebuilt to " << burst_frames_ << " frames";
            }
            last_iaf_ = iaf_;
        }

        TimePoint now = Clock::now();
        TimePoint sample_ts = data_in->source_timestamp();
        // data_out->set_source_timestamp(now);
        data_in_port_->slot(0)->ReleaseData();
        
        double delay_sec = std::chrono::duration<double>(now - sample_ts).count();
        delay_sec += audio_latency_s_() + erp_latency_s_();

        double phase_advance  = 2.0 * M_PI * iaf_ * delay_sec;
        double corrected_phase = std::fmod(phase + phase_advance, 2.0 * M_PI);
        double diff            = corrected_phase - stim_onset_rad_;

        // Wrap diff to [-pi, pi] once
        const double wrapped_diff = std::atan2(std::sin(diff), std::cos(diff));

        output_ = (std::abs(wrapped_diff) < stim_dur_rad_ / 2);

        // Trigger a single burst on the rising edge (false -> true)
        if (output_ && !last_output_) {
            audio_trigger_pending_.store(true);
            // LOG(INFO) << name() << " Packet " << packet_count_ << ": Estimated phase = " << phase << " , delay = " << delay_sec << " s, corr_phase = " << corrected_phase;
        }
        else if (!output_ && last_output_) {
            // LOG(INFO) << name() << " Packet " << packet_count_ << ": Estimated phase = " << phase << " , delay = " << delay_sec << " s, corr_phase = " << corrected_phase;
        }
        last_output_ = output_;

        data_out->set_data_sample(0, 0, static_cast<double>(output_));
        data_out_port_->slot(0)->PublishData();

        packet_count_++;
    }
}

void StimulusController::Postprocess(ProcessingContext &context)
{
    stop_audio_();
    LOG(INFO) << name() << " Total messages processed: " << packet_count_;
}

void StimulusController::Unprepare(GlobalContext &context)
{
    (void)context;
    stop_audio_();
}


REGISTERPROCESSOR(StimulusController);