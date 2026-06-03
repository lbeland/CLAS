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
#include "scalardata/scalardata.hpp"
#include "multichanneldata/multichanneldata.hpp"
#include <gram_savitzky_golay/gram_savitzky_golay.h>
#include <dsp/filter.hpp>
#include <boost/circular_buffer.hpp>
#include <complex>
#include <fftw3.h>
#include <vector>

class IAFEstimator : public IProcessor {
  public:
    IAFEstimator();

  void CreatePorts() override;
  void CompleteStreamInfo() override;
  void Prepare(GlobalContext &context) override;
  void Preprocess(ProcessingContext &context) override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;
  void Unprepare(GlobalContext &context) override;

  // VARIABLES
  protected:
    unsigned int packet_count_ = 0;
    double freq_resolution;
    double fs_ = 0.0;
    size_t n_fft_ = 0;
    size_t window_size_ = 1;
    inline static boost::circular_buffer<float> sample_window{1};  // Initialized with size 1, will be resized in Prepare
    gram_sg::SavitzkyGolayFilter savgol_;
    
    BroadcasterState<double>* iaf_state_ = nullptr;
    double current_iaf_ = std::numeric_limits<double>::quiet_NaN();
    double last_valid_iaf_ = std::numeric_limits<double>::quiet_NaN();
    double current_gauss_width_ = std::numeric_limits<double>::quiet_NaN();
    double ema_mu_;
    double ema_;
    int invalid_count_ = 0;
    int invalid_threshold_;
    int max_analyze_bin;
    int f_min_bin;
    int f_max_bin;
    fftwf_plan fft_plan_;
    float *signal_in;
    fftwf_complex *freq_half;
    std::vector<double> freqs;

    const uint32_t MAX_NCHANNELS=384;

  // DATA PORTS
  protected:
    PortIn<MultiChannelType<float>> *data_in_port_;
    PortOut<ScalarType<double>> *data_out_port_;

  // OPTIONS
  protected:
    options::Int n_messages_{-1};
    options::Double window_size_sec_{5};
    options::Double f_min_{5.0};
    options::Double f_max_{18.0};
    options::Value<unsigned int, false> calc_interval_{100};
    options::Double ema_window_sec_{5.0};
};
    