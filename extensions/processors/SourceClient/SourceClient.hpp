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
  uint32_t trigger_bits;
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
  void Preprocess(ProcessingContext &context) override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;

 protected:
  PortOut<MultiChannelType<float>> *data_out_port_;
  int sock = -1;

  options::Value<int, false> fs_{1000};
  options::Value<int, false> nchannels_{4};
  options::Value<int, false> nsamples_{100};
  options::Value<int, false> n_messages_{10};
  options::Value<std::string, false> output_file_{"SourceClient.csv"};
  
  inline static std::vector<std::chrono::steady_clock::time_point> send_times;
};
