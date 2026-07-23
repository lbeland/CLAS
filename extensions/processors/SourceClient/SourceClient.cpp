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
#include <algorithm>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

constexpr int PORT = 25000;
constexpr size_t PACKET_SIZE = 172;

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
    add_option("recal_interval_s", recal_interval_s_, "Interval (s) between online true-fs re-estimation refits.");
}

void SourceClient::CreatePorts()
{
    data_out_port_ = create_output_port<MultiChannelType<float>>(
        "out",
        MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()),
        PortOutPolicy(SlotRange(2), 200, WaitStrategy::kBlockingStrategy));
}

void SourceClient::CompleteStreamInfo()
{
    // Slot 0: EEG channels
    data_out_port_->slot(0)->streaminfo().set_parameters(MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()));
    data_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());

    // Slot 1: 8 AUX channels + 1 trigger channel (9 total)
    data_out_port_->slot(1)->streaminfo().set_parameters(MultiChannelType<float>::Parameters(9, nsamples_(), fs_()));
    data_out_port_->slot(1)->streaminfo().set_stream_rate(fs_());
}

void SourceClient::Prepare(GlobalContext &context)
{
}

void SourceClient::Preprocess(ProcessingContext &context)
{
    packet_count_ = 0;
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

uint64_t SourceClient::hardware_time_us_(uint64_t sample_counter) const
{
    return anchor_time_us_ + static_cast<uint64_t>(
        static_cast<double>(sample_counter - anchor_n_) * 1e6 / fs_eff_);
}

// Feeds one real (not interpolated) packet observation into the current
// tumbling window, and refits fs_eff_ once recal_interval_s_ has elapsed.
void SourceClient::recalibrate_fs_(uint64_t sample_counter, int64_t ts_us)
{
    if (recal_count_ == 0)
    {
        recal_block_start_n_ = sample_counter;
        recal_block_start_ts_us_ = ts_us;
        recal_next_refit_ts_us_ = ts_us + static_cast<int64_t>(recal_interval_s_() * 1e6);
    }

    const double x = static_cast<double>(sample_counter - recal_block_start_n_);
    const double y = static_cast<double>(ts_us - recal_block_start_ts_us_);

    // Welford's online covariance update.
    ++recal_count_;
    const double dx = x - recal_mean_x_;
    recal_mean_x_ += dx / static_cast<double>(recal_count_);
    recal_mean_y_ += (y - recal_mean_y_) / static_cast<double>(recal_count_);
    recal_cov_xy_ += dx * (y - recal_mean_y_);
    recal_var_x_ += dx * (x - recal_mean_x_);

    if (ts_us < recal_next_refit_ts_us_)
        return;

    if (recal_count_ >= 2 && recal_var_x_ > 0.0)
    {
        const double slope_us_per_sample = recal_cov_xy_ / recal_var_x_;
        const double new_fs_eff = 1e6 / slope_us_per_sample;

        // Real crystal drift is ppm-scale; reject anything further off nominal
        // as a bad fit (e.g. from a packet-loss burst) instead of adopting it.
        const double nominal_fs = fs_();
        if (std::isfinite(new_fs_eff) && std::abs(new_fs_eff - nominal_fs) < 0.005 * nominal_fs)
        {
            anchor_time_us_ = hardware_time_us_(sample_counter);
            anchor_n_ = sample_counter;
            fs_eff_ = new_fs_eff;
            LOG(INFO) << name() << " fs refit: effective sample rate = " << fs_eff_
                      << " Hz (nominal " << nominal_fs << " Hz)";
        }
        else
        {
            LOG(WARNING) << name() << " Rejected implausible fs refit: " << new_fs_eff
                         << " Hz (nominal " << nominal_fs << " Hz); keeping fs_eff_=" << fs_eff_;
        }
    }

    recal_count_ = 0;
    recal_mean_x_ = 0.0;
    recal_mean_y_ = 0.0;
    recal_cov_xy_ = 0.0;
    recal_var_x_ = 0.0;
    recal_block_start_n_ = sample_counter;
    recal_block_start_ts_us_ = ts_us;
    recal_next_refit_ts_us_ = ts_us + static_cast<int64_t>(recal_interval_s_() * 1e6);
}

void SourceClient::Process(ProcessingContext &context)
{
    SlotOut<MultiChannelType<float>> *data_slot = data_out_port_->slot(0);
    SlotOut<MultiChannelType<float>> *aux_slot = data_out_port_->slot(1);
    MultiChannelType<float>::Data *data_out = nullptr;
    MultiChannelType<float>::Data *aux_out = nullptr;
    uint8_t buffer[2048];
    uint32_t first_sample_counter = 0;
    uint32_t sample_counter = 0;

    std::vector<float> eeg_vec(nchannels_());
    std::vector<float> aux_vec(9); // 8 AUX + 1 trigger

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
        const int n_calib = calib_packets_();
        std::vector<int64_t> offsets_us;
        offsets_us.reserve(static_cast<size_t>(std::max(n_calib, 0)));

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
    }

    // Anchor the continuous fs-recalibration mapping at the calibration result.
    // start_time_us_ is defined at relative sample 0 (see calibration loop
    // above), so the anchor must be n=0, not the current sample_counter.
    anchor_time_us_ = start_time_us_;
    anchor_n_ = 0;
    fs_eff_ = fs_();
    recal_count_ = 0;

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
            std::vector<MultiChannelType<float>::Data *> data_out_vec = data_slot->ClaimDataN(missed, false);
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
                std::vector<MultiChannelType<float>::Data *> aux_out_vec = aux_slot->ClaimDataN(missed, false);
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
                    aux_vec[8] = (last_packet_.input_trigger >> 7) & 1 ? 1.0f : 0.0f;
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

        TimePoint after_parsing = Clock::now();

        const uint64_t ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
            timestamp.time_since_epoch()).count();
        recalibrate_fs_(sample_counter, static_cast<int64_t>(ts_us));

        hardware_time_us = hardware_time_us_(sample_counter);
        if (hardware_time_us > ts_us)
        {
            // Calibration floor was set too high for this packet; clamp to now.
            LOG(WARNING) << name() << " hardware_time_us (" << hardware_time_us << ") is in the future (now=" << ts_us << "). Clamping.";
            hardware_time_us = ts_us;
        }
        else if ((int)ts_us - (int)hardware_time_us > 1e6)
        {
            LOG(WARNING) << name() << " hardware_time_us (" << hardware_time_us << ") is more than 1 s behind (now=" << ts_us << ").";
        }

        data_out = data_slot->ClaimData(false);
        TimePoint after_claim = Clock::now();

        std::copy(pkt.eeg.begin(), pkt.eeg.begin() + nchannels_(), eeg_vec.begin());
        data_out->set_data_sample(0, eeg_vec);
        data_out->set_sample_timestamp(0, hardware_time_us + steady_to_wallclock_offset_us_);
        // data_out->set_source_timestamp(micros_to_timepoint(hardware_time_us));
        data_out->set_source_timestamp(Clock::now());

        data_out->set_hardware_timestamp(hardware_time_us + steady_to_wallclock_offset_us_);

        TimePoint after_set_eeg = Clock::now();
        data_slot->PublishData();
        TimePoint after_publish_eeg = Clock::now();

        // AUX + trigger channel
        aux_out = aux_slot->ClaimData(true);
        // aux_out->set_source_timestamp(micros_to_timepoint(hardware_time_us));
        aux_out->set_source_timestamp(Clock::now());
        aux_out->set_hardware_timestamp(hardware_time_us + steady_to_wallclock_offset_us_);
        if (store_aux)
        {
            std::copy(pkt.aux.begin(), pkt.aux.end(), aux_vec.begin());
            aux_vec[8] = (pkt.input_trigger >> 7) & 1 ? 1.0f : 0.0f;
            aux_out->set_data_sample(0, aux_vec);
            aux_out->set_sample_timestamp(0, hardware_time_us + steady_to_wallclock_offset_us_);
        }
        aux_slot->PublishData();
        TimePoint after_publish_aux = Clock::now();

        packet_count_++;
        last_packet_ = pkt;

        TimePoint finished = Clock::now();
        if (std::chrono::duration<double, std::micro>(finished - timestamp).count() > 100)
        {
            LOG(INFO) << name() << " Processed packet " << packet_count_ << " - timings (us):"
                      << " parse=" << std::chrono::duration<double, std::micro>(after_parsing - timestamp).count()
                      << ", claim=" << std::chrono::duration<double, std::micro>(after_claim - after_parsing).count()
                      << ", set_eeg=" << std::chrono::duration<double, std::micro>(after_set_eeg - after_claim).count()
                      << ", pub_eeg=" << std::chrono::duration<double, std::micro>(after_publish_eeg - after_set_eeg).count()
                      << ", pub_aux=" << std::chrono::duration<double, std::micro>(after_publish_aux - after_publish_eeg).count()
                      << ", total=" << std::chrono::duration<double, std::micro>(finished - timestamp).count();
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
