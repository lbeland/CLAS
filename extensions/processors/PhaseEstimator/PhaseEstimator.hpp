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
#include <dsp/filter.hpp>
#include <boost/circular_buffer.hpp>
#include <complex>
#include <fftw3.h>
#include <vector>

class PhaseEstimator : public IProcessor {
  public:
    PhaseEstimator();
    void calibrate_gain(const int N);
    void load_filter_coeffs(const StorageContext& context, double iaf);
    void load_phase_shift(const StorageContext& context, double iaf);

  void CreatePorts() override;
  void CompleteStreamInfo() override;
  void Prepare(GlobalContext &context) override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;
  void Unprepare(GlobalContext &context) override;

  // VARIABLES
  protected:
    unsigned int packet_count_ = 0;
    double first_timestamp_ = 0.0;
    double fs_ = 0.0;
    double f0_ = 10.0; // default IAF value, will be updated from state
    size_t n_fft_ = 0;
    size_t window_size_ = 1;
    inline static boost::circular_buffer<float> sample_window{1};  // Initialized with size 1, will be resized in Preprocess
    std::string coeff_file_;  // Path to bandpass filter coefficients file
    std::vector<std::complex<double>> coeffs_;  // Filter coefficients of bandpass filter
    std::complex<double> c_gain_;  // Calibration gain for cecHT

    std::vector<double> filter_phase_shift_values;
    double filter_phase_shift_ = 0.0;
    
    FollowerState<double>* iaf_state_ = nullptr;
    bool valid_iaf_ = false;

    const uint32_t MAX_NCHANNELS=384;

  // DATA PORTS
  protected:
    PortIn<MultiChannelType<float>> *data_in_port_;
    PortOut<MultiChannelType<double>> *data_out_port_;

  // OPTIONS
  protected:
    options::Value<int, false> n_messages_{-1};
    // options::Value<unsigned int, false> n_fft_{4096};
    options::Value<bool> calibrate_{false};
    options::Value<unsigned int, false> iaf_read_interval_{5000};
    options::Value<YAML::Node, false> filter_def_{};
    options::Value<YAML::Node, false> compensate_filter_{};

};
