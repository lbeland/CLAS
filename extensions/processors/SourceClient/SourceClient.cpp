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
const std::string SOURCE_IP = "192.168.200.21"; // set "" to disable filtering


uint32_t read_u32_le(const uint8_t* p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

float read_f32_le(const uint8_t* p) {
    uint32_t tmp = read_u32_le(p);
    float value;
    std::memcpy(&value, &tmp, sizeof(float));
    return value;
}

bool parse_packet(const uint8_t* data, size_t len, Packet& pkt) {
    if (len < PACKET_SIZE) return false;

    pkt.token = read_u32_le(data + 0);
    pkt.sample_counter = read_u32_le(data + 4);
    pkt.trigger_bits = read_u32_le(data + 8);

    // Aux: 8 floats
    for (size_t i = 0; i < 8; ++i) {
        pkt.aux[i] = read_f32_le(data + 12 + i * 4);
    }

    // EEG: 32 floats
    for (size_t i = 0; i < 32; ++i) {
        pkt.eeg[i] = read_f32_le(data + 44 + i * 4);
    }

    return true;
}

SourceClient::SourceClient() : IProcessor(PRIORITY_HIGH) {
    add_option("fs", fs_, "Sample Frequency of Turbolink Client");
    add_option("nchannels", nchannels_, "Number of channels to generate.");
    add_option("nsamples", nsamples_, "Number of samples per packet.");
    add_option("n_messages", n_messages_, "Number of packets to generate (-1 = infinite).");
    add_option("output_file", output_file_, "Path to output CSV file.");
}

void SourceClient::CreatePorts() {
  data_out_port_ = create_output_port<MultiChannelType<float>>(
      "out",
      MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()),
      PortOutPolicy(SlotRange(1),200,WaitStrategy::kBlockingStrategy));
}

void SourceClient::CompleteStreamInfo() {
  // Set the parameters for the output stream
  dynamic_cast<StreamInfo<MultiChannelType<float>>&>(data_out_port_->slot(0)->streaminfo())
      .set_parameters(MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()));
}

void SourceClient::Preprocess(ProcessingContext &context) {

    send_times.clear();

    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket");
        return;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sock);
        return;
    }

    // Set a receive timeout so recvfrom() does not block forever
    timeval tv{};
    tv.tv_sec = 1;  // 1 second timeout;
    if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        perror("setsockopt");
        close(sock);
        return;
    }

    std::cout << "Listening on UDP port " << PORT << std::endl;
}


void SourceClient::Process(ProcessingContext &context) {
    using clock = std::chrono::steady_clock;
    MultiChannelType<float>::Data *data_out = nullptr;
    uint8_t buffer[2048];
    int first_sample_counter;
    uint32_t sample_counter;
    // Measurement phase
    sockaddr_in src{};
    socklen_t srclen = sizeof(src);
    int packet_count = 0;
    clock::time_point first_timestamp;
    clock::time_point timestamp;

    while (!context.terminated()){
        if (n_messages_() != -1 && packet_count >= n_messages_()) {
            break;
        }

        ssize_t received = recvfrom(sock, buffer, sizeof(buffer), 0,
                                    (struct sockaddr*)&src, &srclen);

        if (received < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // timeout: loop again and check g_running
                continue;
            }

            if (context.terminated()) {
                break;
            }

            perror("recvfrom");
            break;
        }
    
        timestamp = clock::now();

        Packet pkt{};
        if (!parse_packet(buffer, received, pkt)) {
            std::cerr << "Invalid packet size: " << received << std::endl;
            continue;
        }

        // std::cout << "Sample: " << pkt.sample_counter
        //           << " | EEG[0]: " << pkt.eeg[0]
        //           << std::endl;

        if (packet_count == 0) {
            first_sample_counter = pkt.sample_counter;
            sample_counter = pkt.sample_counter;
            first_timestamp = timestamp;
            LOG(INFO) << "\n First packet received. Sample counter: " << sample_counter;
        }
        else {
            if ((pkt.sample_counter > sample_counter+1) || (sample_counter == std::numeric_limits<uint32_t>::max() && pkt.sample_counter != 0)) {
                LOG(WARNING) << "\n Missed packet(s). Last sample counter: " << sample_counter << ", current: " << pkt.sample_counter;
                return;
            }
            else{
                sample_counter = pkt.sample_counter;
            }
            
        }

        // Claim output buffer
        data_out = data_out_port_->slot(0)->ClaimData(false);

        // Send whole eeg array (all channels)
        // data_out->set_data_sample(0, pkt.eeg);

        // Send only as many channels as defined
        for (int i=0;i<nchannels_();i++) {
        float sample = pkt.eeg[i];
        data_out->set_data_sample(0, i, sample);
        }
        
        const auto source_timestamp_us = std::chrono::time_point_cast<std::chrono::microseconds>(timestamp);
        data_out->set_source_timestamp(source_timestamp_us);  // Set source timestamp with microsecond precision

        // Calculate hardware timestamp based on sample counter and fs
        std::chrono::time_point hardware_timestamp = first_timestamp + std::chrono::duration<double>((sample_counter - first_sample_counter)/fs_());
        const uint64_t hw_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(hardware_timestamp.time_since_epoch()).count());

        if (packet_count % 100 == 0) {
            LOG(INFO) << "\n " << name() << ". Received packet " << packet_count + 1 << " with sample " << std::fixed << std::setprecision(2) << pkt.eeg[0] << " with sample counter " << pkt.sample_counter << " (hardware timestamp: " << hw_us << ")";
        }

        data_out->set_hardware_timestamp(hw_us);

        // LOG(INFO) << name() << ". Sent message " << i + 1 << " with sample " << sample << ".";

        // Publish data
        data_out_port_->slot(0)->PublishData();

        send_times.push_back(source_timestamp_us);

        packet_count++;
    }

}

void SourceClient::Postprocess(ProcessingContext &context) {

    std::ostringstream statistic_print;

    // Calculate Statistics
    statistic_print << "\n ---------------- \n Total messages sent: " << send_times.size();

    if (send_times.empty()) {
        return;
    }

    // Calculate statistics
    double sum_diff_us = 0.0;
    double max_diff_us = 0.0;
    std::size_t max_idx = 0;
    double sum_sq_diff = 0.0;
    std::vector<double> send_times_diff;
    send_times_diff.resize(send_times.size() - 1);

    for (std::size_t i = 0; i + 1 < send_times.size(); i++) {
        const double diff_us = std::chrono::duration<double, std::micro>(send_times[i+1] - send_times[i]).count();
        
        send_times_diff[i] = diff_us;
        sum_diff_us += diff_us;
        sum_sq_diff += diff_us * diff_us;
        if (diff_us > max_diff_us) {
            max_diff_us = diff_us;
            max_idx = i;
        }
    }

    const std::size_t n_periods = send_times_diff.size();
    double avg_period = 0.0;
    double std_period = 0.0;
    if (n_periods > 0) {
        avg_period = sum_diff_us / static_cast<double>(n_periods);
        const double variance = (sum_sq_diff / static_cast<double>(n_periods)) - (avg_period * avg_period);
        std_period = sqrt(fmax(0.0, variance));
    }

    statistic_print << "\n Average send period (us): " << avg_period;
    statistic_print << "\n Max send period (us): " << max_diff_us << ", idx: " << max_idx;
    statistic_print << "\n Std send period (us): " << std_period << "\n";

    // Save to CSV
    std::string append = "_SourceClient.csv";
    std::ofstream output;
    output.open(output_file_().c_str() + append);
    output << "Metric,Send_period\n";
    output << "mean," << avg_period << "\n";
    output << "std," << std_period << "\n";
    output << "max," << max_diff_us << "\n";
    output.close();

    append = "_send_times.csv";
    std::ofstream send_times_output;
    send_times_output << std::fixed << std::setprecision(17);
    send_times_output.open(output_file_().c_str() + append);
    for (double t : send_times_diff) {
        send_times_output << t << "\n";
    }
    send_times_output.close();

    std::cout << statistic_print.str();

    close(sock);
    std::cout << "\n Socket closed. Shutdown complete." << std::endl;
}

REGISTERPROCESSOR(SourceClient);
