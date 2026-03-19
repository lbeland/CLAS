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

    // pkt.token = read_u32_le(data + 0);
    pkt.sample_counter = read_u32_le(data + 4);
    // pkt.trigger_bits = read_u32_le(data + 8);

    // // Aux: 8 floats
    // for (size_t i = 0; i < 8; ++i) {
    //     pkt.aux[i] = read_f32_le(data + 12 + i * 4);
    // }

    // // EEG: 32 floats
    // for (size_t i = 0; i < 32; ++i) {
    //     pkt.eeg[i] = read_f32_le(data + 44 + i * 4);
    // }

    return true;
}

SourceClient::SourceClient() : IProcessor(PRIORITY_HIGH) {
  add_option("nchannels", nchannels_, "Number of channels to generate.");
  add_option("nsamples", nsamples_, "Number of samples per packet.");
  add_option("n_messages", n_messages_, "Number of packets to generate (0 = infinite).");
  add_option("output_file", output_file_, "Path to output CSV file.");
}

void SourceClient::CreatePorts() {
  data_out_port_ = create_output_port<MultiChannelType<float>>(
      "out",
      MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), 1.0),
      PortOutPolicy(SlotRange(1),200,WaitStrategy::kBlockingStrategy));
}

void SourceClient::CompleteStreamInfo() {
  // Set the parameters for the output stream
  dynamic_cast<StreamInfo<MultiChannelType<float>>&>(data_out_port_->slot(0)->streaminfo())
      .set_parameters(MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), 1.0));
}

void SourceClient::Preprocess(ProcessingContext &context) {

    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket");
        return;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

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
  MultiChannelType<float>::Data *data_out = nullptr;
  uint8_t buffer[2048];
  int last_sample_counter = -1;
  // Measurement phase
  sockaddr_in src{};
  socklen_t srclen = sizeof(src);

  while (!context.terminated()){

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

    char sender_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &src.sin_addr, sender_ip, sizeof(sender_ip));

    if (!SOURCE_IP.empty() && SOURCE_IP != sender_ip) {
        continue;
    }

    Packet pkt{};
    if (!parse_packet(buffer, received, pkt)) {
        std::cerr << "Invalid packet size: " << received << std::endl;
        continue;
    }

    // std::cout << "Sample: " << pkt.sample_counter
    //           << " | EEG[0]: " << pkt.eeg[0]
    //           << std::endl;
    if (last_sample_counter == -1) {
        last_sample_counter = pkt.sample_counter;
    }
    if (pkt.sample_counter > last_sample_counter+1) {
      printf("\n Warning: Missed packet(s). Last sample counter: %d, current: %d", last_sample_counter, pkt.sample_counter);
      perror("\n Error: Missed packet(s)");
      return;
    }
    else{
      last_sample_counter = pkt.sample_counter;
    }


    // Claim output buffer
    data_out = data_out_port_->slot(0)->ClaimData(false);

    // Send whole eeg array (all channels)
    // data_out->set_data_sample(0, pkt.eeg);

    // Send only as many channels as defined
    for (int i=0;i<nchannels_();i++) {
      auto sample = pkt.eeg[i];
      data_out->set_data_sample(0, i, sample);
    }
    
    data_out->set_source_timestamp();  // Set source timestamp to now

    double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(data_out->source_timestamp().time_since_epoch()).count();
    // printf("%s. Sent message %u with sample %f.\n", name().c_str(), i + 1, sample);

    // Publish data
    data_out_port_->slot(0)->PublishData();

    send_times.push_back(timestamp);

  }

}

void SourceClient::Postprocess(ProcessingContext &context) {

  // Calculate Statistics
  printf("\n Total messages sent: %d", send_times.size());

  if (send_times.empty()) {
      return;
  }

  // Calculate statistics
  double sum_diff = 0.0;
  double max_diff = 0.0;
  int max_idx = 0;
  double sum_sq_diff = 0.0;
  std::vector<double> send_times_diff;
  send_times_diff.resize(send_times.size() - 1);

  for (int i = 0; i < send_times.size()-1; i++) {
      double diff = send_times[i+1] - send_times[i];
      send_times_diff[i] = diff;
      sum_diff += diff;
      sum_sq_diff += diff * diff;
      if (diff > max_diff) {
          max_diff = diff;
          max_idx = i;
      }
  }

  double avg_period_sec = sum_diff / (send_times.size() - 1);
  double avg_period = avg_period_sec * 1e-3;  // us
  double variance = (sum_sq_diff / (send_times.size() - 1)) - (avg_period_sec * avg_period_sec);  // s²
  double std_period = sqrt(fmax(0.0, variance)) * 1e-3;  // us

  printf("\n Average send period (us): %.6f", avg_period);
  printf("\n Max send period (us): %.6f, idx: %d", max_diff * 1e-3, max_idx);
  printf("\n Std send period (us): %.6f\n", std_period);

  // Save to CSV
  std::string append = "_SourceClient.csv";
  std::ofstream output;
  output.open(output_file_().c_str() + append);
  output << "Metric,Send_period\n";
  output << "mean," << avg_period << "\n";
  output << "std," << std_period << "\n";
  output << "max," << max_diff*1e-3 << "\n";
  output.close();

  append = "_send_times.csv";
  std::ofstream send_times_output;
  send_times_output << std::fixed << std::setprecision(17);
  send_times_output.open(output_file_().c_str() + append);
  for (double t : send_times_diff) {
      send_times_output << t << "\n";
  }
  send_times_output.close();

  close(sock);
  std::cout << "\n Socket closed. Shutdown complete." << std::endl;
}

REGISTERPROCESSOR(SourceClient);
