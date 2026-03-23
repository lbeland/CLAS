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
#include <sstream>

Consumer::Consumer() : IProcessor(PRIORITY_HIGH) {
  add_option("n_messages", n_messages_, "Number of packets to receive (0 = infinite).");
  add_option("output_file", output_file_, "Path to output CSV file.");
}

void Consumer::CreatePorts() {
  data_in_port_ = create_input_port<MultiChannelType<float>>(
      "in", 
      MultiChannelType<float>::Capabilities(ChannelRange(1, 256), SampleRange(1, 10000)),
      PortInPolicy(SlotRange(1)));
}

void Consumer::Preprocess(ProcessingContext &context){
  printf("\n");
  const auto& info = data_in_port_->streaminfo(0);
  const auto& p = info.parameters<MultiChannelType<float>::Parameters>();
  printf("Stream parameters - nchannels: %u, nsamples: %u, sample_rate: %f\n", p.nchannels, p.nsamples, p.sample_rate);
  packet_count_ = 0;
  samples.clear();
  recv_times.clear();
  source_times.clear();
}

void Consumer::Process(ProcessingContext &context) {
  MultiChannelType<float>::Data *data_in = nullptr;
  using Clock = std::chrono::steady_clock;
  Clock::time_point receive_timestamp;

  // Measurement phase
  while (!context.terminated()) {
    
    // Try to retrieve data
    if (!data_in_port_->slot(0)->RetrieveData(data_in)) {
      break;
    }
    receive_timestamp = Clock::now();
    auto sample = data_in->data_sample(0,0);  // Get the first sample of the first channel
    auto source_timestamp = data_in->source_timestamp();

    // Release data
    data_in_port_->slot(0)->ReleaseData();

    samples.push_back(sample);
    recv_times.push_back(receive_timestamp);
    source_times.push_back(source_timestamp);

    // printf("%s. Received packet %u with sample %f at %9f\n", name().c_str(), packet_count_ + 1, sample, now);

    packet_count_++;

    // custom_sleep_for(500000);
  }

}

void Consumer::Postprocess(ProcessingContext &context) {

  std::ostringstream statistic_print;
  statistic_print << "\n ---------------- \n Consumer: Total messages processed: " << packet_count_;

  if (packet_count_ == 0) {
      return;
  }

  // Jitter statistics
  double sum_diff_us = 0.0;
  double max_diff_us = 0.0;
  std::size_t max_idx = 0;
  double sum_sq_diff = 0.0;
  std::vector<double> recv_times_diff;
  recv_times_diff.resize(packet_count_ > 0 ? packet_count_ - 1 : 0);

  for (std::size_t i = 0; i + 1 < recv_times.size(); i++) {
    const double diff_us = std::chrono::duration<double, std::micro>(recv_times[i+1] - recv_times[i]).count();
    recv_times_diff[i] = diff_us;
    sum_diff_us += diff_us;
    sum_sq_diff += diff_us * diff_us;
    if (diff_us > max_diff_us) {
      max_diff_us = diff_us;
          max_idx = i;
      }
  }
  const std::size_t n_periods = recv_times_diff.size();
  double avg_period = 0.0;
  double std_period = 0.0;
  if (n_periods > 0) {
    avg_period = sum_diff_us / static_cast<double>(n_periods);
    const double variance = (sum_sq_diff / static_cast<double>(n_periods)) - (avg_period * avg_period);
    std_period = sqrt(fmax(0.0, variance));
  }

  statistic_print << "\n Average receive period (us): " << avg_period;
  statistic_print << "\n Max receive period (us): " << max_diff_us << ", idx: " << max_idx;
  statistic_print << "\n Std receive period (us): " << std_period << "\n";

  // Latency statistics
  double sum_latency_us = 0.0;
  double max_latency_us = 0.0;
  max_idx = 0;
  double sum_sq_latency = 0.0;
  std::vector<double> process_times;
  process_times.resize(packet_count_);

  for (std::size_t i = 0; i < source_times.size(); i++) {
    const double latency_us = std::chrono::duration<double, std::micro>(recv_times[i] - source_times[i]).count();
    process_times[i] = latency_us;
    sum_latency_us += latency_us;
    sum_sq_latency += latency_us * latency_us;
    if (latency_us > max_latency_us) {
      max_latency_us = latency_us;
          max_idx = i;
      }
  }
  const double avg_latency = sum_latency_us / static_cast<double>(packet_count_);
  const double variance_latency = (sum_sq_latency / static_cast<double>(packet_count_)) - (avg_latency * avg_latency);
  const double std_latency = sqrt(fmax(0.0, variance_latency));

  statistic_print << "\n Average latency (us): " << avg_latency;
  statistic_print << "\n Max latency (us): " << max_latency_us << ", idx: " << max_idx;
  statistic_print << "\n Std latency (us): " << std_latency << "\n";

  double throughput = 0.0;
  double elapsed_seconds = 0.0;
  if (packet_count_ > 0) {
    elapsed_seconds = std::chrono::duration<double>(recv_times.back() - source_times.front()).count();
    if (elapsed_seconds > 0.0) {
      throughput = static_cast<double>(packet_count_) / elapsed_seconds;
    }
  }
  statistic_print << "\n Throughput (msg/s): " << packet_count_ << "/" << elapsed_seconds << " = " << throughput;

  std::cout << statistic_print.str() << "\n";


  // Save to CSV
  std::string append = "_Consumer.csv";
  std::ofstream output;
  output.open(output_file_().c_str() + append);
  output << "Metric,Recv_period,Latency,Throughput\n";
  output << "mean," << avg_period << "," << avg_latency << "," << throughput << "\n";
  output << "std," << std_period << "," << std_latency << ",\n";
  output << "max," << max_diff_us << "," << max_latency_us << ",\n";
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

  append = "_samples.csv";
  std::ofstream samples_output;
  samples_output << std::fixed << std::setprecision(17);
  samples_output.open(output_file_().c_str() + append);
  for (double s : samples) {
      samples_output << s << "\n";
  }
  samples_output.close();
}

REGISTERPROCESSOR(Consumer);
