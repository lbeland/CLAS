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

const double PI = 3.141592653589793;

Producer::Producer() : IProcessor(PRIORITY_HIGH) {
  add_option("nchannels", nchannels_, "Number of channels to generate.");
  add_option("nsamples", nsamples_, "Number of samples per packet.");
  add_option("n_messages", n_messages_, "Number of packets to generate (0 = infinite).");
  add_option("output_file", output_file_, "Path to output CSV file.");
}

void Producer::CreatePorts() {
  data_out_port_ = create_output_port<MultiChannelType<double>>(
      "out",
      MultiChannelType<double>::Parameters(nchannels_(), nsamples_(), 1.0),
      PortOutPolicy(SlotRange(1),200,WaitStrategy::kBusySpinStrategy));
}

void Producer::CompleteStreamInfo() {
  // Set the parameters for the output stream
  dynamic_cast<StreamInfo<MultiChannelType<double>>&>(data_out_port_->slot(0)->streaminfo())
      .set_parameters(MultiChannelType<double>::Parameters(nchannels_(), nsamples_(), 1.0));
}

void Producer::Process(ProcessingContext &context) {
  MultiChannelType<double>::Data *data_out = nullptr;
  send_times.resize(n_messages_());

  // Measurement phase
  for (int i=0; i < n_messages_() && !context.terminated(); i++) {
    // Claim output buffer
    data_out = data_out_port_->slot(0)->ClaimData(false);

    // Set timestamp as value
    auto t = i * 0.0001;  // Simulate a sample timestamp (e.g., 10 kHz sample rate)
    std::vector<double> sample(3,sin(2 * PI * t * 10));  // Generate a sine wave with frequency of 10 Hz
    // int sample = i;
    for (int i=0;i<nsamples_();i++) {
      data_out->set_data_sample(i, sample);
    }
    
    data_out->set_source_timestamp();  // Set source timestamp to now

    double timestamp = data_out->source_timestamp().time_since_epoch().count();
    // printf("%s. Sent message %u with sample %f.\n", name().c_str(), i + 1, sample);

    // Publish data
    data_out_port_->slot(0)->PublishData();


    send_times[i] = timestamp;

  }

}

void Producer::Postprocess(ProcessingContext &context) {

  std::ostringstream statistic_print;

  // Calculate Statistics
  statistic_print << "Total messages sent: " << send_times.size() << "\n";

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

  statistic_print << "Average send period (us): " << avg_period << "\n";
  statistic_print << "Max send period (us): " << max_diff * 1e-3 << ", idx: " << max_idx << "\n";
  statistic_print << "Std send period (us): " << std_period << "\n";

  // Save to CSV
  std::string append = "_Producer.csv";
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

  std::cout << statistic_print.str();
}

REGISTERPROCESSOR(Producer);
