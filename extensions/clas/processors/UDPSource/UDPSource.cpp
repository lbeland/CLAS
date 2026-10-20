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

#include "UDPSource.hpp"
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

// Initial RLS covariance: large, unconfident prior around 1/fs_nom
constexpr double CLOCK_RLS_INIT_C = 1;

// Interval (s) between anchor re-basing / fs_eff_ commits
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

UDPSource::UDPSource() : IProcessor(PRIORITY_HIGH)
{
    add_option("fs", fs_, "Sample frequency of Turbolink client (Hz).");
    add_option("nchannels", nchannels_, "Number of EEG channels to receive.");
    add_option("nsamples", nsamples_, "Number of samples per packet.");
    add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
    add_option("store_aux", store_aux_, "Whether to forward auxiliary (AUX + trigger) data.");
    add_option("calib_packets", calib_packets_, "Number of packets used for initial start-time calibration. At 10 kHz, 1000 packets is about 100 ms of startup delay.");
    add_option("fs_tau_s", fs_tau_s_, "Time constant (s) of the forgetting factor for continuous true-fs tracking.");
    add_option("recal_warmup_s", recal_warmup_s_, "Duration (s) of the RLS warm-up: below this, lambda=1 (no forgetting).");
}

void UDPSource::CreatePorts()
{
    eeg_out_port_ = create_output_port<MultiChannelType<double>>(
        "eeg",
        MultiChannelType<double>::Parameters(nchannels_(), nsamples_(), fs_()),
        PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));

    // 8 AUX channels + 1 trigger channel (9 total)
    aux_out_port_ = create_output_port<MultiChannelType<double>>(
        "aux",
        MultiChannelType<double>::Parameters(9, nsamples_(), fs_()),
        PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));
}

void UDPSource::CompleteStreamInfo()
{
    // EEG channels
    eeg_out_port_->slot(0)->streaminfo().set_parameters(MultiChannelType<double>::Parameters(nchannels_(), nsamples_(), fs_()));
    eeg_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());

    // 8 AUX channels + 1 trigger channel (9 total)
    aux_out_port_->slot(0)->streaminfo().set_parameters(MultiChannelType<double>::Parameters(9, nsamples_(), fs_()));
    aux_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());
}

void UDPSource::Prepare(GlobalContext &context)
{
    double tau_s = fs_tau_s_();
    lambda = std::exp(-1.0 / (fs_() * tau_s));
    recal_warmup_samples_ = static_cast<uint64_t>(recal_warmup_s_() * fs_());
    reanchor_interval_samples_ = static_cast<uint64_t>(REANCHOR_INTERVAL_S * fs_());
}

void UDPSource::Preprocess(ProcessingContext &context)
{
    // RLS prior: rate assumed close to nominal, low confidence
    theta1_ = 1e6 / fs_();
    P_ = CLOCK_RLS_INIT_C;

    // Placeholder anchor for the warm-up window, until Process() calibrates it
    fs_eff_ = fs_();
    anchor_n_ = 0;
    anchor_time_us_ = 0;

    packet_count_ = 0;
    // Set for real once the first packet arrives in Process()
    start_time_us_ = 0;

    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_ < 0)
    {
        perror("socket");
        return;
    }

    // Receive timeout so recvfrom() does not block forever
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

// Full-precision prediction, used internally so re-anchoring never loses a
// fractional microsecond
double UDPSource::hardware_time_us_precise_(uint64_t sample_counter) const
{
    return anchor_time_us_ +
        static_cast<double>(sample_counter - anchor_n_) * 1e6 / fs_eff_;
}

// Rounded integer microsecond timestamp for external/output use
uint64_t UDPSource::hardware_time_us_(uint64_t sample_counter) const
{
    return static_cast<uint64_t>(std::llround(hardware_time_us_precise_(sample_counter)));
}

// RLS clock model through the anchor (no intercept), with forgetting factor
// lambda: y(n) = theta1_*x(n), x/y measured relative to the anchor so they
// stay numerically small. lambda=1 during warm-up, then relaxes to track
// slow crystal drift. Re-anchoring (every REANCHOR_INTERVAL_S) keeps x/y
// small without changing theta1_/P_, since this is a through-origin fit.
void UDPSource::recalibrate_fs_(uint64_t sample_counter, int64_t ts_us)
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

    // Only re-anchor once every reanchor_interval_samples_ packets
    if (x < static_cast<double>(reanchor_interval_samples_))
        return;

    // Reject implausible fits (packet loss, noise spikes); crystal drift is ppm-scale
    const double nominal_fs = fs_();
    const double new_fs_eff = 1e6 / theta1_;
    if (std::isfinite(new_fs_eff) && std::abs(new_fs_eff - nominal_fs) < 0.00001 * nominal_fs)
    {
        // Extrapolate the anchor with the newly fit rate, not the stale one
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

void UDPSource::Process(ProcessingContext &context)
{
    SlotOut<MultiChannelType<double>> *data_slot = eeg_out_port_->slot(0);
    SlotOut<MultiChannelType<double>> *aux_slot = aux_out_port_->slot(0);
    MultiChannelType<double>::Data *data_out = nullptr;
    MultiChannelType<double>::Data *aux_out = nullptr;
    uint8_t buffer[2048];
    uint32_t first_sample_counter = 0;
    uint32_t sample_counter = 0;

    // Samples arrive as float32, mapped to double for downstream processors
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

    // Calibration: receive calib_packets_ packets without publishing them, and
    // estimate start_time_us_ from the minimum observed (timestamp - ideal
    // sample time) offset, since reception jitter only ever delays a packet
    {
        const int n_calib = std::max(calib_packets_(), 1); // 0 would skip the loop below entirely

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

        // Anchor at (start_time_us_, sample 0) now that calibration is done
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
            // Calibration floor was set too high for this packet; clamp to now
            hardware_time_us = ts_us;
        }
        else if (ts_us - hardware_time_us > 5 * 1e3)
        {
            LOG(WARNING) << name() << " hardware_time_us (" << hardware_time_us << ") is more than 5ms behind (now=" << ts_us << "). Clamping";
            hardware_time_us = ts_us - 5 * 1e3;
        }

        data_out = data_slot->ClaimData(false);

        std::copy(pkt.eeg.begin(), pkt.eeg.begin() + nchannels_(), eeg_vec.begin());
        data_out->set_data_sample(0, eeg_vec);
        data_out->set_sample_timestamp(0, hardware_time_us + steady_to_wallclock_offset_us_);
        // Receival time, not hardware_time_us, so downstream latency can be measured
        data_out->set_source_timestamp(Clock::now());

        data_out->set_hardware_timestamp(hardware_time_us + steady_to_wallclock_offset_us_);

        data_slot->PublishData();

        // AUX + trigger channel
        aux_out = aux_slot->ClaimData(true);
        // Receival time, not hardware_time_us, so downstream latency can be measured
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

void UDPSource::Postprocess(ProcessingContext &context)
{
    close(sock_);
    LOG(INFO) << name() << ": Socket closed. Total packets received: " << packet_count_;
}

void UDPSource::Unprepare(GlobalContext &context)
{
}

REGISTERPROCESSOR(UDPSource);
