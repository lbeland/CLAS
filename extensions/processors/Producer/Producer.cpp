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

#include "Producer.hpp"
#include "logging/log.hpp"
#include "threadutilities.hpp"
#include <thread>
#include <chrono>
#include <cmath>
#include <sstream>
#include <array>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>

namespace
{

    struct SignalState
    {
        double value;
        double amplitude;
        double theta;
        double inst_freq;
    };

    double WrapPhase(double phase)
    {
        return std::remainder(phase, 2.0 * M_PI);
    }

    // Incremental Voss-McCartney 1/f ("pink") noise generator. Same algorithm family
    // as StimulusController's block-based voss() (Downey, ThinkDSP); this variant is
    // stateful so it can be pulled one sample at a time inside the streaming loop.
    class PinkNoise
    {
    public:
        explicit PinkNoise(std::uint32_t seed, int nrows = 16)
            : rng_(seed), nrows_(std::clamp(nrows, 1, 24)),
              index_mask_((1u << nrows_) - 1u), rows_(static_cast<size_t>(nrows_), 0.0)
        {
            for (double &r : rows_)
            {
                r = dist_(rng_);
                running_sum_ += r;
            }
        }

        // Returns roughly 1/f-distributed noise; raw variance ~ (nrows + 1) / 3.
        double next()
        {
            index_ = (index_ + 1u) & index_mask_;
            if (index_ != 0u)
            {
                int row = 0;
                for (unsigned int n = index_; (n & 1u) == 0u; n >>= 1)
                    ++row;
                running_sum_ -= rows_[static_cast<size_t>(row)];
                const double v = dist_(rng_);
                running_sum_ += v;
                rows_[static_cast<size_t>(row)] = v;
            }
            return running_sum_ + dist_(rng_);
        }

    private:
        std::mt19937 rng_;
        std::uniform_real_distribution<double> dist_{-1.0, 1.0};
        int nrows_;
        unsigned int index_mask_;
        unsigned int index_ = 0u;
        std::vector<double> rows_;
        double running_sum_ = 0.0;
    };

    SignalState ComputeSignalState(double carrier_phase,
                                   double carrier_amplitude,
                                   double carrier_frequency,
                                   const std::string &modulation_type,
                                   double modulation_amplitude,
                                   double modulation_frequency,
                                   double modulation_phase)
    {
        SignalState state{};

        if (modulation_type == "amplitude")
        {
            state.amplitude = carrier_amplitude + modulation_amplitude * std::cos(modulation_phase);
            state.theta = carrier_phase;
            state.inst_freq = carrier_frequency;
            state.value = state.amplitude * std::cos(state.theta);
            return state;
        }

        if (modulation_type == "phase")
        {
            if (modulation_frequency == 0.0)
            {
                state.amplitude = carrier_amplitude;
                state.theta = carrier_phase + modulation_amplitude;
                state.inst_freq = carrier_frequency;
                state.value = state.amplitude * std::cos(state.theta);
                return state;
            }

            state.amplitude = carrier_amplitude;
            state.theta = carrier_phase + (modulation_amplitude / modulation_frequency) * std::sin(modulation_phase);
            state.inst_freq = carrier_frequency + modulation_amplitude * std::cos(modulation_phase);
            state.value = state.amplitude * std::cos(state.theta);
            return state;
        }

        // Default: no modulation
        state.amplitude = carrier_amplitude;
        state.theta = carrier_phase;
        state.inst_freq = carrier_frequency;
        state.value = state.amplitude * std::cos(state.theta);
        return state;
    }

} // namespace

Producer::Producer() : IProcessor(PRIORITY_HIGH)
{
    add_option("path", path_, "Path (server-side) where to save data.");
    add_option("fs", fs_, "Sample frequency (Hz).");
    add_option("carrier_amplitude", carrier_amplitude_, "Carrier signal amplitude.");
    add_option("carrier_frequency", carrier_frequency_, "Carrier signal frequency in Hz.");
    add_option("modulation_type", modulation_type_,
               "Mutually exclusive signal mode: none | amplitude | phase | phase_jump | "
               "freq_step | chirp.");
    add_option("modulation_amplitude", modulation_amplitude_,
               "amplitude/phase mode: modulation depth.");
    add_option("modulation_frequency", modulation_frequency_,
               "amplitude/phase mode: modulation rate (Hz).");
    add_option("mod_period_s", mod_period_s_,
               "phase_jump / freq_step mode: length (s) of each unperturbed segment. The "
               "perturbation toggles on/off at every multiple of this period.");
    add_option("phase_jump_angle", phase_jump_angle_,
               "phase_jump mode: phase offset (rad) applied while toggled on.");
    add_option("freq_step_size", freq_step_size_,
               "freq_step mode: frequency offset (Hz) added to the carrier while toggled on.");
    add_option("chirp_slope", chirp_slope_,
               "chirp mode: linear carrier-frequency slope (Hz/s), applied continuously from t=0.");
    add_option("noise_color", noise_color_,
               "Additive noise colour: white | pink (Voss-McCartney 1/f). Independent per channel.");
    add_option("snr_db", snr_db_,
               "In-band SNR in dB: carrier power (carrier_amplitude^2/2) vs noise power measured "
               "within a band of width noise_band_hz centred on carrier_frequency. White noise is "
               "scaled analytically; pink noise uses an ideal 1/f model for the in-band fraction.");
    add_option("noise_band_hz", noise_band_hz_,
               "Width (Hz) of the band around carrier_frequency in which snr_db is defined.");
    add_option("burst_on_s", burst_on_s_,
               "Bursty signal: duration (s) of each ON segment.");
    add_option("burst_off_s", burst_off_s_,
               "Bursty signal: duration (s) of each OFF segment. 0 = signal always on. "
               "During OFF segments the signal is silenced (noise only) and the ground-truth "
               "meta output is set to NaN.");
    add_option("nchannels", nchannels_, "Number of channels to generate.");
    add_option("nsamples", nsamples_, "Number of samples per packet.");
    add_option("n_messages", n_messages_, "Number of packets to generate (-1 = infinite).");
    add_option("inter_packet_sleep", inter_packet_sleep_,
               "When true (default), sleep between packets so output is paced at the real-time "
               "sample rate. When false, generate packets as fast as possible with no "
               "inter-packet sleep.");

    iaf_state_ = create_broadcaster_state<double>(
        "iaf", current_iaf_, Permission::NONE,
        "Individual alpha frequency shared with downstream processors.");
}

void Producer::CreatePorts()
{
    data_out_port_ = create_output_port<MultiChannelType<double>>(
        "out",
        MultiChannelType<double>::Parameters(nchannels_(), nsamples_(), fs_()),
        PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));

    meta_out_port_ = create_output_port<MultiChannelType<double>>(
        "meta_out",
        MultiChannelType<double>::Parameters(3, 1, fs_()),
        PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));
}

void Producer::CompleteStreamInfo()
{
    data_out_port_->slot(0)->streaminfo().set_parameters(
        MultiChannelType<double>::Parameters(nchannels_(), nsamples_(), fs_()));
    data_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());

    meta_out_port_->slot(0)->streaminfo().set_parameters(
        MultiChannelType<double>::Parameters(3, 1, fs_()));
    meta_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());
}

void Producer::Preprocess(ProcessingContext &context)
{
    packet_count_ = 0;
}

void Producer::Process(ProcessingContext &context)
{
    MultiChannelType<double>::Data *data_out = nullptr;
    MultiChannelType<double>::Data *meta_out = nullptr;

    std::vector<double> sample_vec(nchannels_());

    double carrier_phase = 0.0;
    double modulation_phase = 0.0;
    uint64_t hardware_time_us = 0;

    const double modulation_step = 2.0 * M_PI * modulation_frequency_() / fs_();

    // ---- Additive noise, scaled for a target in-band SNR relative to the carrier ----
    // snr_db := 10*log10( P_carrier / P_noise_in_band ), where P_noise_in_band is the noise
    // power within [carrier_frequency +/- noise_band_hz/2].
    std::mt19937 noise_rng{std::random_device{}()};
    std::normal_distribution<double> white_dist(0.0, 1.0);

    const bool pink_noise = (noise_color_() == "pink");
    const double nyquist = 0.5 * fs_();
    const double carrier_power = 0.5 * carrier_amplitude_() * carrier_amplitude_();
    const double band_lo = std::max(1e-6, carrier_frequency_() - 0.5 * noise_band_hz_());
    const double band_hi = std::min(nyquist, carrier_frequency_() + 0.5 * noise_band_hz_());
    const double band_width = std::max(1e-9, band_hi - band_lo);
    const double target_inband_power = carrier_power * std::pow(10.0, -snr_db_() / 10.0);

    // noise_gain multiplies the unit output of the chosen generator so that its in-band
    // power equals target_inband_power.
    double noise_gain = 0.0;
    std::vector<PinkNoise> pink_gen;
    if (pink_noise)
    {
        constexpr int kPinkRows = 16;
        constexpr int kCalN = 1 << 15;

        // Empirical total power (sample variance) of the raw generator output.
        PinkNoise cal(noise_rng(), kPinkRows);
        double s = 0.0, s2 = 0.0;
        for (int i = 0; i < kCalN; ++i)
        {
            const double v = cal.next();
            s += v;
            s2 += v * v;
        }
        const double raw_var =
            std::max(1e-12, s2 / kCalN - (s / kCalN) * (s / kCalN));

        // In-band fraction from an ideal 1/f model with a low-frequency knee at
        // fs / 2^rows (below which Voss-McCartney flattens); total power ~ 1 + ln(fN/fknee).
        const double f_knee = nyquist / std::pow(2.0, kPinkRows);
        const double total_power_norm = 1.0 + std::log(nyquist / f_knee);
        const double lo = std::max(band_lo, f_knee);
        const double inband_frac =
            std::max(1e-12, std::log(band_hi / lo) / total_power_norm);

        noise_gain = std::sqrt(target_inband_power / inband_frac) / std::sqrt(raw_var);

        pink_gen.reserve(nchannels_());
        for (unsigned int c = 0; c < nchannels_(); ++c)
            pink_gen.emplace_back(noise_rng(), kPinkRows);
    }
    else
    {
        // White noise: flat one-sided PSD = var / nyquist, so the power in a band of
        // width band_width is var * band_width / nyquist.
        noise_gain = std::sqrt(target_inband_power * nyquist / band_width);
    }

    // Bursty signal: alternate ON/OFF segments. OFF => signal silenced, ground truth = NaN.
    const std::string mode = modulation_type_();
    const double burst_cycle = burst_on_s_() + burst_off_s_();
    const bool bursting = burst_off_s_() > 0.0 && burst_cycle > 0.0;
    const double kNaN = std::numeric_limits<double>::quiet_NaN();

    // Base time in steady_clock (same domain as source_timestamp / Clock::now()).
    // The wallclock offset converts it to UTC µs for hardware_timestamp, matching SourceClient.
    uint64_t start_time = std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now().time_since_epoch()).count();
    TimePoint start_time_point = Clock::now();
    int64_t steady_to_wallclock_offset_us =
        static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) -
        static_cast<int64_t>(start_time);

    // Expected wall-clock gap between consecutive packets, used only to pace emission.
    const double inter_packet_ns = (static_cast<double>(nsamples_()) / fs_()) * 1e9;

    // last_emit_time_ = Clock::now();

    while (!context.terminated())
    {
        if (n_messages_() != -1 && packet_count_ >= n_messages_())
        {
            break;
        }

        // const TimePoint target_emit_time =
        //     last_emit_time_ + std::chrono::nanoseconds(static_cast<int64_t>(inter_packet_ns));
        const TimePoint target_emit_time = start_time_point + packet_count_ * std::chrono::nanoseconds(static_cast<int64_t>(inter_packet_ns));

        if (inter_packet_sleep_())
        {
            const TimePoint now = Clock::now();
            if (target_emit_time > now)
            {
                constexpr int64_t puffer = 5;
                const int64_t remaining_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    target_emit_time - now).count();
                const int64_t sleep_for = std::max<int64_t>(remaining_us - puffer, 0);
                custom_sleep_for(static_cast<uint64_t>(sleep_for));
            }
        }

        // Elapsed signal time (consistent with per-packet phase/timestamp advance).
        const double t = static_cast<double>(packet_count_) / fs_();

        // --- non-stationary modes (mutually exclusive with amplitude/phase modulation) ---
        // phase_jump / freq_step alternate: toggled on during every other period segment.
        double effective_freq = carrier_frequency_();
        double phase_offset = 0.0;
        const bool toggled = mod_period_s_() > 0.0 &&
                             (static_cast<long long>(std::floor(t / mod_period_s_())) % 2 != 0);
        if (mode == "chirp")
        {
            effective_freq = carrier_frequency_() + chirp_slope_() * t;
        }
        else if (mode == "freq_step" && toggled)
        {
            effective_freq = carrier_frequency_() + freq_step_size_();
        }
        else if (mode == "phase_jump" && toggled)
        {
            phase_offset = phase_jump_angle_();
        }

        SignalState state = ComputeSignalState(
            carrier_phase + phase_offset,
            carrier_amplitude_(),
            effective_freq,
            mode,
            modulation_amplitude_(),
            modulation_frequency_(),
            modulation_phase);

        // Bursty signal: silence the signal and NaN the ground truth during OFF segments.
        bool signal_on = true;
        if (bursting)
        {
            signal_on = std::fmod(t, burst_cycle) < burst_on_s_();
        }

        std::vector<double> meta_data;
        if (signal_on)
        {
            meta_data = {state.amplitude, state.theta, state.inst_freq};
            current_iaf_ = state.inst_freq;
            iaf_state_->set(current_iaf_);
        }
        else
        {
            state.value = 0.0;  // only white noise remains
            meta_data = {kNaN, kNaN, kNaN};
            // iaf broadcaster holds its last valid value for downstream processors.
        }

        // hardware_time_us is in steady_clock µs; convert to wall-clock for hardware_timestamp
        hardware_time_us = start_time + (uint64_t)packet_count_ * 1000000ULL / fs_();

        // Independent noise per channel (white Gaussian or pink 1/f), scaled for in-band SNR.
        for (unsigned int c = 0; c < nchannels_(); ++c)
        {
            const double n = pink_noise ? pink_gen[c].next() : white_dist(noise_rng);
            sample_vec[c] = state.value + noise_gain * n;
        }

        data_out = data_out_port_->slot(0)->ClaimData(false);

        data_out->set_data_sample(0, sample_vec);
        data_out->set_sample_timestamp(0, hardware_time_us + steady_to_wallclock_offset_us);
        data_out->set_source_timestamp(TimePoint(std::chrono::microseconds(hardware_time_us)));
        data_out->set_hardware_timestamp(hardware_time_us + steady_to_wallclock_offset_us);
        data_out_port_->slot(0)->PublishData();

        meta_out = meta_out_port_->slot(0)->ClaimData(false);
        meta_out->set_data_sample(0, meta_data);
        meta_out->set_sample_timestamp(0, hardware_time_us + steady_to_wallclock_offset_us);
        meta_out->set_source_timestamp(TimePoint(std::chrono::microseconds(hardware_time_us)));
        meta_out->set_hardware_timestamp(hardware_time_us + steady_to_wallclock_offset_us);
        meta_out_port_->slot(0)->PublishData();

        ++packet_count_;

        // Re-anchor from the actual emit time (not the fixed schedule), so the next
        // iteration's pacing reflects reality rather than trying to catch up.
        // last_emit_time_ = Clock::now();

        // Advance the carrier phase with the (possibly perturbed) instantaneous frequency
        // so that frequency steps and chirps integrate continuously into the phase.
        carrier_phase = WrapPhase(carrier_phase + 2.0 * M_PI * effective_freq / fs_());
        modulation_phase = WrapPhase(modulation_phase + modulation_step);
    }
    LOG(INFO) << name() << " stopped working";
}

void Producer::Postprocess(ProcessingContext &context)
{
    LOG(INFO) << name() << ": Total messages sent: " << packet_count_;
}

REGISTERPROCESSOR(Producer);
