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
#pragma once

#include "iprocessor.hpp"
#include "multichanneldata/multichanneldata.hpp"
#include "options/options.hpp"

struct Packet {
  uint32_t token;
  uint32_t sample_counter;
  uint8_t input_trigger;
      std::vector<float> aux = std::vector<float>(8);
      std::vector<float> eeg = std::vector<float>(32);
};

uint32_t read_u32_le(const uint8_t* p);
float read_f32_le(const uint8_t* p);
bool parse_packet(const uint8_t* data, size_t len, Packet& pkt);

class SourceClient : public IProcessor {
 public:
  SourceClient();

  void CreatePorts() override;
  void CompleteStreamInfo() override;
  void Prepare(GlobalContext &context) override;
  void Preprocess(ProcessingContext &context) override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;
  void Unprepare(GlobalContext &context) override;

 protected:
  PortOut<MultiChannelType<float>> *data_out_port_;
  int sock = -1;
  bool audio_aux_ = false;
  int packet_count_ = 0;
  Packet last_packet_;

  options::Double fs_{1000};
  options::Int nchannels_{32};  // 32 EEG
  options::Int nsamples_{100};
  options::Int n_messages_{-1};
  options::Bool store_aux_{true};

  // Number of packets used for the initial start-time calibration.
  // At 10 kHz, 1000 packets = 100 ms of startup delay.
  options::Int calib_packets_{1000};
 
  // Calibrated reference time (microseconds, in Clock's epoch) such that
  // hardware_time_us(n) = start_time_us_ + n * 1e6 / fs approximates the
  // true ADC sampling time of sample n (plus the fixed transit delay floor).
  // Stored as a member so a later re-calibration step can update it.
  uint64_t start_time_us_ = 0;
 
  // Offset (microseconds) such that:
  //   wallclock_us = clock_us + steady_to_wallclock_offset_us_
  // Sampled once during calibration so hardware_time_us values (which live
  // in Clock's/steady epoch) can be converted back to a real datetime
  // during post-hoc analysis. Not used in any in-loop latency math.
  int64_t steady_to_wallclock_offset_us_ = 0;
 
//   std::vector<std::chrono::steady_clock::time_point> send_times;

};