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

uint32_t read_u32_le(const uint8_t *p);
float    read_f32_le(const uint8_t *p);
bool     parse_packet(const uint8_t *data, size_t len, Packet &pkt);

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
    // Data ports
    PortOut<MultiChannelType<float>> *data_out_port_;

    // Options
    options::Double fs_{1000};
    options::Int nchannels_{32};
    options::Int nsamples_{100};
    options::Int n_messages_{-1};
    options::Bool store_aux_{true};
    // Number of packets used for initial start-time calibration.
    // At 10 kHz, 1000 packets ≈ 100 ms of startup delay.
    options::Int calib_packets_{10000};
    // Time constant [s] of the exponential forgetting factor
    options::Double fs_tau_s_{15.0};
    // Duration [s] of the RLS warm-up
    options::Double recal_warmup_s_{30.0};

    // Runtime state
    int sock_ = -1;
    int packet_count_ = 0;
    Packet last_packet_;

    // Calibrated reference time [µs, in Clock's epoch] such that
    // hardware_time_us(n) = start_time_us_ + n * 1e6 / fs approximates
    // the true ADC sampling time of sample n (plus the fixed transit delay floor).
    uint64_t start_time_us_ = 0;

    // Offset [µs] such that: wallclock_us = clock_us + steady_to_wallclock_offset_us_
    // Sampled once during calibration to allow post-hoc conversion from
    // steady-clock hardware timestamps back to wall-clock time.
    int64_t steady_to_wallclock_offset_us_ = 0;

    // fs_eff_ tracks crystal drift
    double fs_eff_ = 0.0;
    uint64_t anchor_n_ = 0;
    uint64_t anchor_time_us_ = 0;

    double P_ = 0.0;
    double theta_slope_ = 0.0;
    double lambda = 1.0;
    uint64_t recal_warmup_samples_ = 0;

    bool recal_anchor_set_ = false;
    uint64_t recal_anchor_n_ = 0;
    int64_t recal_anchor_ts_us_ = 0;

    uint64_t hardware_time_us_(uint64_t sample_counter) const;
    void recalibrate_fs_(uint64_t sample_counter, int64_t ts_us);
};
