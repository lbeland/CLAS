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
#include <fstream>
#include <iomanip>
#include "threadutilities.hpp"
#include <thread>
#include <chrono>
#include <cmath>
#include <sstream>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

constexpr int PORT = 25000;
constexpr size_t PACKET_SIZE = 172;
// const std::string SOURCE_IP = "192.168.200.21"; // set "" to disable filtering

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
    pkt.input_trigger = static_cast<uint8_t>(trigger_bits & 0xFF);  // first 8 bits are input, next 8 bits are output

    // Aux: 8 floats
    for (size_t i = 0; i < 8; ++i)
    {
        pkt.aux[i] = read_f32_le(data + 12 + i * 4);
    }

    // EEG: 32 floats
    for (size_t i = 0; i < 32; ++i)
    {
        pkt.eeg[i] = read_f32_le(data + 44 + i * 4);
    }

    return true;
}

SourceClient::SourceClient() : IProcessor(PRIORITY_HIGH)
{
    add_option("fs", fs_, "Sample Frequency of Turbolink Client");
    add_option("nchannels", nchannels_, "Number of channels to generate.");
    add_option("nsamples", nsamples_, "Number of samples per packet.");
    add_option("n_messages", n_messages_, "Number of packets to generate (-1 = infinite).");
    add_option("store_aux", store_aux_, "Whether to store auxiliary data.");
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
    // Set the parameters for the EEG output stream
    data_out_port_->slot(0)->streaminfo().set_parameters(MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()));
    data_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());

    // Set the parameters for the AUX and Trigger output stream
    data_out_port_->slot(1)->streaminfo().set_parameters(MultiChannelType<float>::Parameters(9, nsamples_(), fs_()));   
    data_out_port_->slot(1)->streaminfo().set_stream_rate(fs_());
}

void SourceClient::Prepare(GlobalContext &context)
{

}

void SourceClient::Preprocess(ProcessingContext &context)
{
    // send_times.clear();
    packet_count_ = 0;

    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0)
    {
        perror("socket");
        return;
    }

    // Set a receive timeout so recvfrom() does not block forever
    timeval tv{};
    tv.tv_sec = 1; // 1 second timeout;
    if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0)
    {
        LOG(ERROR) << name() << " Failed to set socket receive timeout:" << strerror(errno);
        close(sock);
        return;
    }

    LOG(INFO) << name() << "Listening on UDP port " << PORT << std::endl;
}

void SourceClient::Process(ProcessingContext &context)
{
    SlotOut<MultiChannelType<float>>* data_slot = data_out_port_->slot(0);
    SlotOut<MultiChannelType<float>>* aux_slot = data_out_port_->slot(1);
    MultiChannelType<float>::Data *data_out = nullptr;
    MultiChannelType<float>::Data *aux_out = nullptr;
    uint8_t buffer[2048];
    uint32_t first_sample_counter = 0;
    uint32_t sample_counter = 0;

    std::vector<float> eeg_vec(nchannels_());
    std::vector<float> aux_vec(9); // 8 aux channels + 1

    // Measurement phase
    sockaddr_in src{};
    socklen_t srclen = sizeof(src);
    TimePoint first_timestamp;
    TimePoint timestamp;
    uint64_t hardware_time_us = 0;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    ssize_t received;
    bool store_aux = store_aux_();
    Packet pkt{};

    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        LOG(ERROR) << name() << " Failed to bind socket:" << strerror(errno);
        // perror("bind");
        close(sock);
        LOG(ERROR) << name() << " Failed to bind socket. Processor will not receive data.";
        return;
    }

    // Use wall clock time as reference for hardware timestamps
    uint64_t start_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    while (!context.terminated())
    {
        if (n_messages_() != -1 && packet_count_ >= n_messages_())
        {
            break;
        }

        received = recvfrom(sock, buffer, sizeof(buffer), 0,
                                    (struct sockaddr *)&src, &srclen);
        timestamp = Clock::now();

        if (received < 0)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                // timeout: loop again and check g_running
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

        if (packet_count_ == 0)
        {
            first_sample_counter = pkt.sample_counter;
            sample_counter = pkt.sample_counter - first_sample_counter;
            first_timestamp = timestamp;
            LOG(INFO) << name() << " First packet received. Sample counter: " << first_sample_counter;
        }
        else
        {
            if (((pkt.sample_counter - first_sample_counter) > sample_counter + 1))
            {
                int missed = (pkt.sample_counter - first_sample_counter) - sample_counter;
                LOG(ERROR) << name() << " Missed packet(s). Last sample counter: " << sample_counter << ", current: " << pkt.sample_counter - first_sample_counter << ". Missed " << missed << " packets.";
                
                int virt_sample_counter = sample_counter;
                std::vector<MultiChannelType<float>::Data *> data_out_vec = data_slot->ClaimDataN(missed, false);
                for (auto &data_out : data_out_vec)
                {
                    virt_sample_counter++;
                    hardware_time_us = start_time + (uint64_t)virt_sample_counter * 1000000ULL / fs_();
                    // Linear interpolation (set last stored packet)
                    std::copy(last_packet_.eeg.begin(), last_packet_.eeg.begin() + nchannels_(), eeg_vec.begin());
                    data_out->set_data_sample(0, eeg_vec);

                    data_out->set_sample_timestamp(0, hardware_time_us);
                    data_out->set_source_timestamp(timestamp);
                    data_out->set_hardware_timestamp(hardware_time_us);
                }
                // data_slot->PublishData();

                virt_sample_counter = sample_counter;
                if (store_aux)
                {
                    std::vector<MultiChannelType<float>::Data *> aux_out_vec = aux_slot->ClaimDataN(missed, false);
                    for (auto &aux_out : aux_out_vec)              
                    {
                        virt_sample_counter++;
                        hardware_time_us = start_time + (uint64_t)virt_sample_counter * 1000000ULL / fs_();
                        std::copy(last_packet_.aux.begin(), last_packet_.aux.end(), aux_vec.begin());
                        aux_vec[8] = (last_packet_.input_trigger >> 7) & 1 ? 1.0f : 0.0f; // Add trigger data in ninth channel
                        aux_out->set_data_sample(0, aux_vec);

                        // LOG(INFO) << name() << " Packet count: " << packet_count << " Received trigger: " << std::bitset<8>(pkt.input_trigger) << ", " << (pkt.input_trigger >> 7);
                        aux_out->set_sample_timestamp(0, hardware_time_us);
                        aux_out->set_source_timestamp(timestamp);
                        aux_out->set_hardware_timestamp(hardware_time_us);
                    }
                    // aux_slot->PublishData();
                }
                sample_counter += missed;
            }
            else
            {
                sample_counter += 1;
            }
        }

        TimePoint after_parsing = Clock::now();

        hardware_time_us = start_time + (uint64_t)sample_counter * 1000000ULL / fs_();
        // if (packet_count % 100 == 0) {
        //     LOG(INFO) << name() << ". Received packet " << packet_count + 1 << " with sample " << std::fixed << std::setprecision(2) << pkt.eeg[0] << " with sample counter " << pkt.sample_counter << " (hardware timestamp: " << hw_us << ")";
        // }

        data_out = data_slot->ClaimData(false);

        TimePoint after_claim = Clock::now();

        // Send only as many channels as defined
        std::copy(pkt.eeg.begin(), pkt.eeg.begin() + nchannels_(), eeg_vec.begin());
        data_out->set_data_sample(0, eeg_vec);        

        data_out->set_sample_timestamp(0, hardware_time_us);
        data_out->set_source_timestamp(timestamp);
        data_out->set_hardware_timestamp(hardware_time_us);

        // LOG(INFO) << name() << ". Sent message " << packet_count + 1 << " with sample " << pkt.eeg[0] << ".";

        TimePoint after_set_eeg = Clock::now();

        // Publish data
        data_slot->PublishData();

        TimePoint after_publish_eeg = Clock::now();

        // Handle AUX and Trigger data if defined
        if (store_aux)
        {
            aux_out = aux_slot->ClaimData(false);
            std::copy(pkt.aux.begin(), pkt.aux.end(), aux_vec.begin());
            aux_vec[8] = (pkt.input_trigger >> 7) & 1 ? 1.0f : 0.0f; // Add trigger data in ninth channel
            aux_out->set_data_sample(0, aux_vec);

            aux_out->set_sample_timestamp(0, hardware_time_us);
            // LOG(INFO) << name() << " Packet count: " << packet_count << " Received trigger: " << std::bitset<8>(pkt.input_trigger) << ", " << (pkt.input_trigger >> 7);
            aux_out->set_sample_timestamp(0, hardware_time_us);
            aux_out->set_source_timestamp(timestamp);
            aux_out->set_hardware_timestamp(hardware_time_us);
            aux_slot->PublishData();
        }

        TimePoint after_publish_aux = Clock::now();

        // if ((pkt.input_trigger << 7) & 1)
        // {
        //     LOG(INFO) << name() << " Packet count: " << packet_count << " Received input trigger: " << std::bitset<8>(pkt.input_trigger);
        // }

        // send_times.push_back(timestamp);

        packet_count_++;
        last_packet_ = pkt;

        TimePoint finished = Clock::now();
        if (std::chrono::duration<double, std::micro>(finished - timestamp).count() > 100)
        {
            LOG(INFO) << name() << " Processed packet " << packet_count_ << " - timings (us):" 
                      << " parse_packet=" << std::chrono::duration<double, std::micro>(after_parsing - timestamp).count()
                      << ", after_claim=" << std::chrono::duration<double, std::micro>(after_claim - after_parsing).count()
                      << ", after_set_eeg=" << std::chrono::duration<double, std::micro>(after_set_eeg - after_claim).count()
                      << ", after_publish_eeg=" << std::chrono::duration<double, std::micro>(after_publish_eeg - after_set_eeg).count()
                      << ", after_publish_aux=" << std::chrono::duration<double, std::micro>(after_publish_aux - after_publish_eeg).count()
                      << ", total=" << std::chrono::duration<double, std::micro>(finished - timestamp).count();
            // LOG(INFO) << name() << "Processed packet in " << std::chrono::duration<double, std::micro>(finished - timestamp).count() << " us.";
        }
    }
    LOG(INFO) << name() << " stopped working";

}

void SourceClient::Postprocess(ProcessingContext &context)
{
    close(sock);
    LOG(INFO) << name() << " Socket closed.";

    // std::ostringstream statistic_print;

    // // Calculate Statistics
    // statistic_print << "\n ---------------- \n Total messages sent: " << send_times.size();

    // if (send_times.empty())
    // {
    //     return;
    // }

    // // Calculate statistics
    // double sum_diff_us = 0.0;
    // double max_diff_us = 0.0;
    // std::size_t max_idx = 0;
    // double sum_sq_diff = 0.0;
    // std::vector<double> send_times_diff;
    // send_times_diff.resize(send_times.size() - 1);

    // for (std::size_t i = 0; i + 1 < send_times.size(); i++)
    // {
    //     const double diff_us = std::chrono::duration<double, std::micro>(send_times[i + 1] - send_times[i]).count();

    //     send_times_diff[i] = diff_us;
    //     sum_diff_us += diff_us;
    //     sum_sq_diff += diff_us * diff_us;
    //     if (diff_us > max_diff_us)
    //     {
    //         max_diff_us = diff_us;
    //         max_idx = i;
    //     }
    // }

    // const std::size_t n_periods = send_times_diff.size();
    // double avg_period = 0.0;
    // double std_period = 0.0;
    // if (n_periods > 0)
    // {
    //     avg_period = sum_diff_us / static_cast<double>(n_periods);
    //     const double variance = (sum_sq_diff / static_cast<double>(n_periods)) - (avg_period * avg_period);
    //     std_period = sqrt(fmax(0.0, variance));
    // }

    // statistic_print << "\n Average send period (us): " << avg_period;
    // statistic_print << "\n Max send period (us): " << max_diff_us << ", idx: " << max_idx;
    // statistic_print << "\n Std send period (us): " << std_period << "\n";

    // std::cout << statistic_print.str();

}

void SourceClient::Unprepare(GlobalContext &context)
{

}

REGISTERPROCESSOR(SourceClient);
