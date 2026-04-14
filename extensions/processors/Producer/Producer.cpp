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
#include <fstream>
#include <iomanip>
#include "threadutilities.hpp"
#include <thread>
#include <chrono>
#include <cmath>
#include <sstream>

const double PI = std::acos(-1.0);

Producer::Producer() : IProcessor(PRIORITY_HIGH) {
  add_option("fs", fs_, "Sample Frequency");
  add_option("f0", f0_, "Frequency of the sine wave to generate.");
  add_option("nchannels", nchannels_, "Number of channels to generate.");
  add_option("nsamples", nsamples_, "Number of samples per packet.");
  add_option("n_messages", n_messages_, "Number of packets to generate (-1 = infinite).");
  add_option("window_size", window_size_, "Window size of PhaseEstimator (exclude the first window_size packets in statistics calculation).");
  add_option("output_file", output_file_, "Path to output CSV file.");
}

void Producer::CreatePorts() {
  data_out_port_ = create_output_port<MultiChannelType<float>>(
      "out",
      MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()),
      PortOutPolicy(SlotRange(0,MAX_NCHANNELS),200,WaitStrategy::kBusySpinStrategy));
}

void Producer::CompleteStreamInfo() {
  // Set the parameters for the output stream
  data_out_port_->slot(0)->streaminfo().set_parameters(MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()));
  data_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());
}

void Producer::Process(ProcessingContext &context) {
  MultiChannelType<float>::Data *data_out = nullptr;

  send_times.clear();
  float sample = 0.0;
  float t = 0.0;
  int packet_count = 0;

  // Measurement phase
  while (!context.terminated()) {
    if (n_messages_() != -1 && packet_count >= n_messages_()) {
      break;
    }
    // Claim output buffer
    data_out = data_out_port_->slot(0)->ClaimData(false);


    // Set timestamp as value
    t = packet_count * (1.0 / fs_());  // Simulate a sample timestamp (e.g., 10 kHz sample rate)
    sample = 1*sin(2 * PI * t * f0_());  // + 0.2*sin(2 * PI * t * 40) + 0.2*sin(2 * PI * t * 100);  // Generate a sine wave with frequency of 10 Hz
    // int sample = i;
    for (int i=0;i<nchannels_();i++) {
      data_out->set_data_sample(0, i, sample);
    }
    
    const auto source_timestamp_us = std::chrono::time_point_cast<std::chrono::microseconds>(Clock::now());
        
    data_out->set_source_timestamp(source_timestamp_us);  // Set source timestamp to now
    // const uint64_t hw_us = static_cast<uint64_t>(t*1e6);;

    data_out->set_hardware_timestamp(packet_count);

    // LOG(INFO) << name() << ". Sent message " << packet_count + 1 << " with sample " << sample << ".";

    // Publish data
    data_out_port_->slot(0)->PublishData();

    send_times.push_back(source_timestamp_us);
    packet_count++;

  }
}

void Producer::Postprocess(ProcessingContext &context) {

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
  int start_idx = window_size_();  // Skip the first window_size packets because PhaseEstimator is not activ when window is not full
  int n_times = send_times.size() - start_idx;
  if (n_times <= 0) {
      statistic_print << "\n Not enough messages to calculate statistics after skipping the first " << window_size_() << " packets.";
      std::cout << statistic_print.str();
      return;
  }

  send_times_diff.resize(static_cast<std::size_t>(n_times-1));

  for (std::size_t i = start_idx; i + 1 < send_times.size(); i++) {
      const double diff_us = std::chrono::duration<double, std::micro>(send_times[i+1] - send_times[i]).count();
      
      send_times_diff[i-start_idx] = diff_us;
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
  std::string append = "_Producer.csv";
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
}

REGISTERPROCESSOR(Producer);
