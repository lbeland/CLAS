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

#include "Consumer.hpp"
#include "logging/log.hpp"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <limits>

Consumer::Consumer() : IProcessor(PRIORITY_HIGH) {
  add_option("n_messages", n_messages_, "Number of packets to receive (0 = infinite).");
  add_option("output_file", output_file_, "Path to output CSV file.");
}

void Consumer::CreatePorts() {
  data_in_port_ = create_input_port<MultiChannelType<double>>(
      "in", 
      MultiChannelType<double>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
      PortInPolicy(SlotRange(1)));
}

void Consumer::Preprocess(ProcessingContext &context){
  printf("\n");
  const auto& info = data_in_port_->streaminfo(0);
  const auto& p = info.parameters<MultiChannelType<double>::Parameters>();
  printf("Stream parameters - nchannels: %u, nsamples: %u, sample_rate: %f\n", p.nchannels, p.nsamples, p.sample_rate);
  // recv_times.resize(n_messages_());
  // process_times.resize(n_messages_());
}

void Consumer::Process(ProcessingContext &context) {
  MultiChannelType<double>::Data *data_in = nullptr;

  // Measurement phase
  while (packet_count_ < n_messages_() && !context.terminated()) {
    
    // Try to retrieve data
    if (!data_in_port_->slot(0)->RetrieveData(data_in)) {
      break;
    }
    // Release data
    data_in_port_->slot(0)->ReleaseData();
    double now = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();

    // Process the data, take first sample of first channel
    double timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(data_in->source_timestamp().time_since_epoch()).count();
    if (packet_count_ == 0) {
      first_timestamp_ = timestamp;
    }

    recv_times.push_back(now);
    process_times.push_back(now - timestamp);

    // printf("%s. Received packet %u with timestamp %9f at %9f\n", name().c_str(), packet_count_ + 1, timestamp, now);
    
    packet_count_++;

    // custom_sleep_for(500000);
  }

}

void Consumer::Postprocess(ProcessingContext &context) {

  printf("Total messages processed: %d\n", packet_count_);

  // Jitter statistics
  double sum_diff = 0.0;
  double max_diff = 0.0;
  int max_idx = 0;
  double sum_sq_diff = 0.0;
  std::vector<double> recv_times_diff;
  recv_times_diff.resize(packet_count_ - 1);

  for (int i = 0; i < packet_count_-1; i++) {
      double diff = recv_times[i+1] - recv_times[i];
      recv_times_diff[i] = diff;
      sum_diff += diff;
      sum_sq_diff += diff * diff;
      if (diff > max_diff) {
          max_diff = diff;
          max_idx = i;
      }
  }
  double avg_period_ns = sum_diff / (packet_count_ - 1);
  double avg_period = avg_period_ns * 1e-3;  // us
  double variance = (sum_sq_diff / (packet_count_ - 1)) - (avg_period_ns * avg_period_ns);  // s²
  double std_period = sqrt(fmax(0.0, variance)) * 1e-3;  // us

  printf("Average receive period (us): %.6f\n", avg_period);
  printf("Max receive period (us): %.6f, idx: %d\n", max_diff*1e-3, max_idx);
  printf("Std receive period (us): %.6f\n", std_period);

  // Latency statistics
  double sum = 0.0;
  double max = 0.0;
  max_idx = 0;
  double sum_sq = 0.0;

  for (int i=0;i<packet_count_;i++) {
      sum += process_times[i];
      sum_sq += process_times[i] * process_times[i];
      if (process_times[i] > max) {
          max = process_times[i];
          max_idx = i;
      }
  }
  double avg_latency_sec = sum / packet_count_;
  double avg_latency = avg_latency_sec * 1e-3;  // us
  double variance_latency = (sum_sq / packet_count_) - (avg_latency_sec * avg_latency_sec);  // s²
  double std_latency = sqrt(fmax(0.0, variance_latency)) * 1e-3;  // us

  printf("Average latency (us): %.6f\n", avg_latency);
  printf("Max latency (us): %.6f, idx: %d\n", max*1e-3, max_idx);
  printf("Std latency (us): %.6f\n", std_latency);

  double throughput = packet_count_ / (recv_times[packet_count_-1] - first_timestamp_)/1e-6;
  printf("Throughput (msg/ms): %d/%.6f = %.2f\n", packet_count_, (recv_times[packet_count_-1] - first_timestamp_)*1e-6, throughput);


  // Save to CSV
  std::string append = "_Consumer.csv";
  std::ofstream output;
  output.open(output_file_().c_str() + append);
  output << "Metric,Recv_period,Latency,Throughput\n";
  output << "mean," << avg_period << "," << avg_latency << "," << throughput << "\n";
  output << "std," << std_period << "," << std_latency << ",\n";
  output << "max," << max_diff*1e-3 << "," << max*1e-3 << ",\n";
  output.close();

  append = "_recv_times.csv";
  std::ofstream recv_times_output;
  recv_times_output << std::fixed << std::setprecision(17);
  recv_times_output.open(output_file_().c_str() + append);
  for (double t : recv_times_diff) {
      recv_times_output << t << "\n";
  }
  recv_times_output.close();

  append = "_process_times.csv";
  std::ofstream process_times_output;
  process_times_output << std::fixed << std::setprecision(17);
  process_times_output.open(output_file_().c_str() + append);
  for (double t : process_times) {
      process_times_output << t << "\n";
  }
  process_times_output.close();
}

REGISTERPROCESSOR(Consumer);
