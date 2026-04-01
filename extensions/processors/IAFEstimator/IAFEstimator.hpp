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

class IAFEstimator : public IProcessor {
  public:
    IAFEstimator();
    int get_max_bin(fftwf_complex* out, size_t out_size);
    void construct_analytic_spectrum(int M, const fftwf_complex* half, fftwf_complex* full);
    void fftshift(const fftwf_complex* in, fftwf_complex* out, int L);
    void ifftshift(const fftwf_complex* in, fftwf_complex* out, int L);
    void ifftshift(const std::vector<std::complex<float>>& in, std::vector<std::complex<float>>& out, int L);
    void calibrate_gain(const int N);
    void load_filter_coeffs();

  void Configure(const GlobalContext &context) override;
  void CreatePorts() override;
  void CompleteStreamInfo() override;
  void Preprocess(ProcessingContext &context) override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;

  // VARIABLES
  protected:
    unsigned int packet_count_ = 0;
    double first_timestamp_ = 0.0;
    float fs_ = 0.0;
    float f0_ = 0.0;
    inline static boost::circular_buffer<float> sample_window{1};  // Initialized with size 1, will be resized in Preprocess
    std::string coeff_file_;  // Path to bandpass filter coefficients file
    std::vector<std::complex<float>> coeffs_;  // Filter coefficients of bandpass filter
    std::complex<float> c_gain_;  // Calibration gain for cecHT

    const uint32_t MAX_NCHANNELS=384;

  // DATA PORTS
  protected:
    PortIn<MultiChannelType<float>> *data_in_port_;
    PortOut<MultiChannelType<float>> *data_out_port_;

  // OPTIONS
  protected:
    options::Value<int, false> n_messages_{-1};
    options::Value<unsigned int, false> n_fft_{4096};
    options::Value<bool> calibrate_{false};
    options::Value<float, false> iaf_default_{10.0f};
    options::Value<unsigned int, false> iaf_read_interval_{256};
    options::Value<YAML::Node, false> filter_def_{};

    FollowerState<float>* iaf_state_ = nullptr;
};
