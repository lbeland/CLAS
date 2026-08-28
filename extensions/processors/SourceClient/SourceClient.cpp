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

#include "SourceClient.hpp"
#include "logging/log.hpp"
#include "threadutilities.hpp"
#include <thread>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

constexpr int PORT = 25000;
constexpr size_t PACKET_SIZE = 172;

// P(n0) = C: initial RLS covariance scale, i.e. a large, poorly confident
// prior around the 1/fs_nom initial guess.
constexpr double CLOCK_RLS_INIT_C = 1;

// Interval [s] between anchor re-basing / fs_eff_ commits
constexpr double REANCHOR_INTERVAL_S = 10.0;

uint32_t read_u32_le(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

float read_f32_le(const uint8_t *p)
{
    uint32_t tmp = read_u32_le(p);
    float value;
    std::memcpy(&value, &tmp, sizeof(float));
    return value;
}

bool parse_packet(const uint8_t *data, size_t len, Packet &pkt)
{
    if (len < PACKET_SIZE)
        return false;

    pkt.token = read_u32_le(data + 0);
    pkt.sample_counter = read_u32_le(data + 4);
    uint32_t trigger_bits = read_u32_le(data + 8);
    pkt.input_trigger = static_cast<uint8_t>(trigger_bits & 0xFF); // first 8 bits are input, next 8 are output

    for (size_t i = 0; i < 8; ++i)
    {
        pkt.aux[i] = read_f32_le(data + 12 + i * 4);
    }

    for (size_t i = 0; i < 32; ++i)
    {
        pkt.eeg[i] = read_f32_le(data + 44 + i * 4);
    }

    return true;
}

namespace
{
    // Convert a microsecond count expressed in Clock's epoch back into a TimePoint.
    TimePoint micros_to_timepoint(uint64_t us)
    {
        return TimePoint(std::chrono::microseconds(us));
    }
}

SourceClient::SourceClient() : IProcessor(PRIORITY_HIGH)
{
    add_option("fs", fs_, "Sample frequency of Turbolink client (Hz).");
    add_option("nchannels", nchannels_, "Number of EEG channels to receive.");
    add_option("nsamples", nsamples_, "Number of samples per packet.");
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("store_aux", store_aux_, "Whether to forward auxiliary (AUX + trigger) data.");
    add_option("calib_packets", calib_packets_, "Number of packets used for initial start-time calibration.");
    add_option("fs_tau_s", fs_tau_s_, "Time constant (s) of the forgetting factor for continuous true-fs tracking.");
    add_option("recal_warmup_s", recal_warmup_s_, "Duration (s) of the RLS warm-up: below this, lambda=1 (no forgetting).");
}

void SourceClient::CreatePorts()
{
    data_out_port_ = create_output_port<MultiChannelType<double>>(
        "out",
        MultiChannelType<double>::Parameters(nchannels_(), nsamples_(), fs_()),
        PortOutPolicy(SlotRange(2), 200, WaitStrategy::kBlockingStrategy));
}

void SourceClient::CompleteStreamInfo()
{
    // Slot 0: EEG channels
    data_out_port_->slot(0)->streaminfo().set_parameters(MultiChannelType<double>::Parameters(nchannels_(), nsamples_(), fs_()));
    data_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());

    // Slot 1: 8 AUX channels + 1 trigger channel (9 total)
    data_out_port_->slot(1)->streaminfo().set_parameters(MultiChannelType<double>::Parameters(9, nsamples_(), fs_()));
    data_out_port_->slot(1)->streaminfo().set_stream_rate(fs_());
}

void SourceClient::Prepare(GlobalContext &context)
{
    double tau_s = fs_tau_s_();
    lambda = std::exp(-1.0 / (fs_() * tau_s));
    recal_warmup_samples_ = static_cast<uint64_t>(recal_warmup_s_() * fs_());
    reanchor_interval_samples_ = static_cast<uint64_t>(REANCHOR_INTERVAL_S * fs_());
}

void SourceClient::Preprocess(ProcessingContext &context)
{
    // theta1_(n0) = 1/fs_nom: the true rate is assumed close to nominal.
    // P(n0) = C: a poorly confident prior.
    theta1_ = 1e6 / fs_();
    P_ = CLOCK_RLS_INIT_C;

    // Anchor starts at n0 with the nominal fs; both are only meaningful
    // once calibration in Process() has run, but need a defined value for
    // the warm-up window (see hardware_time_us_()).
    fs_eff_ = fs_();
    anchor_n_ = 0;
    anchor_time_us_ = 0;

    packet_count_ = 0;
    // n0/start_time_us_ themselves can only be set once the first packet
    // arrives (see Process()); reset to 0 here for a clean Postprocess/
    // Preprocess cycle between recordings.
    start_time_us_ = 0;

    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_ < 0)
    {
        perror("socket");
        return;
    }

    // Set a receive timeout so recvfrom() does not block forever
    timeval tv{};
    tv.tv_sec = 1;
    if (setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0)
    {
        LOG(ERROR) << name() << " Failed to set socket receive timeout: " << strerror(errno);
        close(sock_);
        return;
    }

    LOG(INFO) << name() << " Listening on UDP port " << PORT;
}

// Full-precision, unrounded prediction: used internally (re-anchoring) so
// that no fractional microsecond is ever discarded from the running state.
double SourceClient::hardware_time_us_precise_(uint64_t sample_counter) const
{
    return anchor_time_us_ +
        static_cast<double>(sample_counter - anchor_n_) * 1e6 / fs_eff_;
}

// Rounded (not truncated) integer microsecond timestamp for external/output
// use. Rounding here is a one-off at the point of emission - it does not
// feed back into anchor_time_us_, so it cannot accumulate across calls.
uint64_t SourceClient::hardware_time_us_(uint64_t sample_counter) const
{
    return static_cast<uint64_t>(std::llround(hardware_time_us_precise_(sample_counter)));
}

// Recursive Least Squares (RLS) clock model, fit through the anchor (no
// intercept term), with exponential forgetting factor lambda:
//   y(n) = theta1_ * x(n) + xi(n),
//   x(n) = sample_counter - anchor_n_, y(n) = t(n) - anchor_time_us_,
//   updated in O(1):
//
//   K(n)      = P(n-1) x(n) / (lambda + x(n)^2 P(n-1))
//   theta1(n) = theta1(n-1) + K(n) [y(n) - theta1(n-1) x(n)]
//   P(n)      = [P(n-1) - K(n) x(n) P(n-1)] / lambda
//
// lambda = 1 during warm-up (no forgetting, every sample weighted
// equally, so P shrinks as fast as possible), then relaxes to the
// fs_tau_s_-derived steady-state forgetting factor, letting the fit track
// slow crystal drift.
//
// x(n)/y(n) are anchor-relative rather than measured from n0, so that x(n)
// stays numerically small between re-anchors instead of growing for the
// rest of the recording. theta1_/P_ are exactly invariant to shifting the
// anchor (this is a through-origin fit), so re-basing loses no information
// - anchor_time_us_ is kept as a double (hardware_time_us_precise_(), not
// the rounded hardware_time_us_()) precisely so re-basing never discards a
// fractional microsecond. Re-basing is still only committed every
// REANCHOR_INTERVAL_S seconds (below) rather than every packet, simply to
// bound how often the RLS commits a new fs_eff_/anchor.
void SourceClient::recalibrate_fs_(uint64_t sample_counter, int64_t ts_us)
{
    const double x = static_cast<double>(sample_counter - anchor_n_);
    const double y = static_cast<double>(ts_us) - anchor_time_us_;

    const bool warming_up = sample_counter < recal_warmup_samples_;
    const double effective_lambda = warming_up ? 1.0 : lambda;

    const double K = P_ * x / (effective_lambda + x * x * P_);
    const double error = y - theta1_ * x;
    theta1_ += K * error;
    P_ = (P_ - K * x * P_) / effective_lambda;

    if (warming_up)
        return;

    // Let x(n) accumulate until the next scheduled re-anchor instead of
    // committing a new anchor/fs_eff_ on every packet.
    if (x < static_cast<double>(reanchor_interval_samples_))
        return;

    // Crystal drift is ppm-scale; reject implausible fits (packet-loss
    // bursts, noise spikes) instead of adopting them into the anchor used
    // for scheduling - keep the last accepted fs_eff_/anchor instead.
    const double nominal_fs = fs_();
    const double new_fs_eff = 1e6 / theta1_;
    if (std::isfinite(new_fs_eff) && std::abs(new_fs_eff - nominal_fs) < 0.00001 * nominal_fs)
    {
        // Update fs_eff_ *before* using it to place the anchor: the anchor
        // must be extrapolated with the rate that was just fit over this
        // interval (theta1_/new_fs_eff), not the stale rate committed at
        // the previous re-anchor - otherwise every commit silently jumps
        // the anchor off the fitted line by (new_fs_eff vs old fs_eff_)
        // worth of drift accumulated over the whole interval.
        fs_eff_ = new_fs_eff;
        anchor_time_us_ = hardware_time_us_precise_(sample_counter);
        anchor_n_ = sample_counter;
    }
    else
    {
        LOG(DEBUG) << name() << " Rejected implausible fs update: " << new_fs_eff
                    << " Hz (nominal " << nominal_fs << " Hz); keeping fs_eff_=" << fs_eff_;
    }
}

void SourceClient::Process(ProcessingContext &context)
{
    SlotOut<MultiChannelType<double>> *data_slot = data_out_port_->slot(0);
    SlotOut<MultiChannelType<double>> *aux_slot = data_out_port_->slot(1);
    MultiChannelType<double>::Data *data_out = nullptr;
    MultiChannelType<double>::Data *aux_out = nullptr;
    uint8_t buffer[2048];
    uint32_t first_sample_counter = 0;
    uint32_t sample_counter = 0;

    // Samples arrive over UDP as float32 (see parse_packet/read_f32_le); they are
    // mapped to double immediately here so every downstream processor works in double.
    std::vector<double> eeg_vec(nchannels_());
    std::vector<double> aux_vec(9); // 8 AUX + 1 trigger

    sockaddr_in src{};
    socklen_t srclen = sizeof(src);
    TimePoint timestamp;
    uint64_t hardware_time_us = 0;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    ssize_t received;
    bool store_aux = store_aux_();
    Packet pkt{};

    if (bind(sock_, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        LOG(ERROR) << name() << " Failed to bind socket: " << strerror(errno);
        close(sock_);
        LOG(ERROR) << name() << " Processor will not receive data.";
        return;
    }

    // -----------------------------------------------------------------
    // Calibration phase
    //
    // Receive calib_packets_ packets without publishing them. Use the
    // (timestamp, sample_counter) pairs to estimate start_time_us_ such
    // that:
    //   hardware_time_us(n) = start_time_us_ + n * 1e6 / fs
    // approximates the true ADC sampling time of sample n (plus the fixed
    // transit-delay floor). Because reception jitter is one-sided (packets
    // can only be delayed, never early), the minimum observed offset is the
    // best estimate of that floor.
    // -----------------------------------------------------------------
    {
        // At least 1: a value of 0 would skip the receive loop below
        // entirely, leaving first_sample_counter/sample_counter at their
        // stale defaults for the main loop.
        const int n_calib = std::max(calib_packets_(), 1);
        std::vector<int64_t> offsets_us;
        offsets_us.reserve(static_cast<size_t>(n_calib));

        bool have_first = false;
        int collected = 0;
        steady_to_wallclock_offset_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() -
            std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now().time_since_epoch()).count();

        while (collected < n_calib && !context.terminated())
        {
            received = recvfrom(sock_, buffer, sizeof(buffer), 0,
                                (struct sockaddr *)&src, &srclen);
            timestamp = Clock::now();

            if (received < 0)
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                    continue;
                if (context.terminated())
                    break;
                perror("recvfrom");
                break;
            }

            if (!parse_packet(buffer, received, pkt))
            {
                std::cerr << "Invalid packet size during calibration: " << received << std::endl;
                continue;
            }

            if (!have_first)
            {
                first_sample_counter = pkt.sample_counter;
                have_first = true;
            }

            const int64_t ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                timestamp.time_since_epoch()).count();
            const int64_t rel_samples = static_cast<int64_t>(pkt.sample_counter) -
                                        static_cast<int64_t>(first_sample_counter);
            const int64_t ideal_us = static_cast<int64_t>(
                static_cast<double>(rel_samples) * 1e6 / fs_());

            offsets_us.push_back(ts_us - ideal_us);

            last_packet_ = pkt;
            collected++;
        }

        if (!offsets_us.empty())
        {
            start_time_us_ = static_cast<uint64_t>(
                *std::min_element(offsets_us.begin(), offsets_us.end()));

            sample_counter = pkt.sample_counter - first_sample_counter;

            LOG(INFO) << name() << " Calibration complete using " << collected
                      << " packets. start_time_us_=" << start_time_us_
                      << ", first_sample_counter=" << first_sample_counter;
        }
        else
        {
            start_time_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now().time_since_epoch()).count();
            LOG(WARNING) << name() << " Calibration received no packets; using current time as fallback start_time_us_=" << start_time_us_;
        }

        // Anchor hardware_time_us_()'s post-warm-up projection at n0 =
        // (start_time_us_, sample 0), now that calibration has produced the
        // real start_time_us_ (Preprocess() can only zero-initialize it).
        fs_eff_ = fs_();
        anchor_n_ = 0;
        anchor_time_us_ = start_time_us_;
    }

    while (!context.terminated())
    {
        if (n_messages_() != -1 && packet_count_ >= n_messages_())
        {
            break;
        }

        received = recvfrom(sock_, buffer, sizeof(buffer), 0,
                            (struct sockaddr *)&src, &srclen);
        timestamp = Clock::now();

        if (received < 0)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                continue;
            }
            if (context.terminated())
            {
                break;
            }
            perror("recvfrom");
            break;
        }

        if (!parse_packet(buffer, received, pkt))
        {
            std::cerr << "Invalid packet size: " << received << std::endl;
            continue;
        }

        // Sequence check: detect and fill dropped packets
        if (((pkt.sample_counter - first_sample_counter) > sample_counter + 1))
        {
            int missed = (pkt.sample_counter - first_sample_counter) - sample_counter;
            LOG(WARNING) << name() << " Missed " << missed << " packet(s). Last counter: " << sample_counter << ", current: " << pkt.sample_counter - first_sample_counter;

            int virt_sample_counter = sample_counter;
            std::vector<MultiChannelType<double>::Data *> data_out_vec = data_slot->ClaimDataN(missed, false);
            for (auto &data_out : data_out_vec)
            {
                virt_sample_counter++;
                hardware_time_us = hardware_time_us_(virt_sample_counter);
                const uint64_t ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    timestamp.time_since_epoch()).count();
                if (hardware_time_us > ts_us)
                {
                    hardware_time_us = ts_us;
                }
                // Fill gap with the last known packet values
                std::copy(last_packet_.eeg.begin(), last_packet_.eeg.begin() + nchannels_(), eeg_vec.begin());
                data_out->set_data_sample(0, eeg_vec);
                data_out->set_sample_timestamp(0, hardware_time_us + steady_to_wallclock_offset_us_);
                data_out->set_source_timestamp(micros_to_timepoint(hardware_time_us));
                data_out->set_hardware_timestamp(hardware_time_us + steady_to_wallclock_offset_us_);
            }

            virt_sample_counter = sample_counter;
            if (store_aux)
            {
                std::vector<MultiChannelType<double>::Data *> aux_out_vec = aux_slot->ClaimDataN(missed, false);
                for (auto &aux_out : aux_out_vec)
                {
                    virt_sample_counter++;
                    hardware_time_us = hardware_time_us_(virt_sample_counter);
                    const uint64_t ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        timestamp.time_since_epoch()).count();
                    if (hardware_time_us > ts_us)
                    {
                        hardware_time_us = ts_us;
                    }
                    std::copy(last_packet_.aux.begin(), last_packet_.aux.end(), aux_vec.begin());
                    aux_vec[8] = (last_packet_.input_trigger >> 7) & 1 ? 1.0 : 0.0;
                    aux_out->set_data_sample(0, aux_vec);
                    aux_out->set_sample_timestamp(0, hardware_time_us + steady_to_wallclock_offset_us_);
                    aux_out->set_source_timestamp(micros_to_timepoint(hardware_time_us));
                    aux_out->set_hardware_timestamp(hardware_time_us + steady_to_wallclock_offset_us_);
                }
            }
            sample_counter += missed;
        }
        else
        {
            sample_counter += 1;
        }

        const uint64_t ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
            timestamp.time_since_epoch()).count();
        recalibrate_fs_(sample_counter, static_cast<int64_t>(ts_us));

        hardware_time_us = hardware_time_us_(sample_counter);
        if (hardware_time_us > ts_us)
        {
            // Calibration floor was set too high for this packet; clamp to now.
            // LOG(WARNING) << name() << " hardware_time_us (" << hardware_time_us << ") is in the future (now=" << ts_us << "). Clamping.";
            hardware_time_us = ts_us;
        }
        else if ((int)ts_us - (int)hardware_time_us > 5 * 1e3)
        {
            LOG(WARNING) << name() << " hardware_time_us (" << hardware_time_us << ") is more than 5ms behind (now=" << ts_us << "). Clamping";
            hardware_time_us = ts_us - 5 * 1e3;
        }

        data_out = data_slot->ClaimData(false);

        std::copy(pkt.eeg.begin(), pkt.eeg.begin() + nchannels_(), eeg_vec.begin());
        data_out->set_data_sample(0, eeg_vec);
        data_out->set_sample_timestamp(0, hardware_time_us + steady_to_wallclock_offset_us_);
        // data_out->set_source_timestamp(micros_to_timepoint(hardware_time_us));
        data_out->set_source_timestamp(Clock::now());

        data_out->set_hardware_timestamp(hardware_time_us + steady_to_wallclock_offset_us_);

        data_slot->PublishData();

        // AUX + trigger channel
        aux_out = aux_slot->ClaimData(true);
        // aux_out->set_source_timestamp(micros_to_timepoint(hardware_time_us));
        aux_out->set_source_timestamp(Clock::now());
        aux_out->set_hardware_timestamp(hardware_time_us + steady_to_wallclock_offset_us_);
        if (store_aux)
        {
            std::copy(pkt.aux.begin(), pkt.aux.end(), aux_vec.begin());
            aux_vec[8] = (pkt.input_trigger >> 7) & 1 ? 1.0 : 0.0;
            aux_out->set_data_sample(0, aux_vec);
            aux_out->set_sample_timestamp(0, hardware_time_us + steady_to_wallclock_offset_us_);
        }
        aux_slot->PublishData();

        packet_count_++;
        last_packet_ = pkt;

        TimePoint finished = Clock::now();
        if (std::chrono::duration<double, std::micro>(finished - timestamp).count() > 100)
        {
            LOG(INFO) << name() << " Processed packet " << packet_count_ << " - timings (us):"
                      << ", total=" << std::chrono::duration<double, std::micro>(finished - timestamp).count();
        }
        if (packet_count_ % static_cast<int>(10 * fs_()) == 0)
        {
            LOG(INFO) << name() << " processed packet " << packet_count_ << ", fs_eff_="
                      << std::fixed << std::setprecision(4) << fs_eff_;
        }
    }
    LOG(INFO) << name() << " stopped working";
}

void SourceClient::Postprocess(ProcessingContext &context)
{
    close(sock_);
    LOG(INFO) << name() << ": Socket closed. Total packets received: " << packet_count_;
}

void SourceClient::Unprepare(GlobalContext &context)
{
}

REGISTERPROCESSOR(SourceClient);
