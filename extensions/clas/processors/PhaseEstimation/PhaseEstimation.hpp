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

class PhaseEstimation : public IProcessor {
  public:
    PhaseEstimation();

    void load_filter_coeffs(const StorageContext &context, double f0);
    void load_phase_shift(const StorageContext &context, double f0);

    void CreatePorts() override;
    void CompleteStreamInfo() override;
    void Prepare(GlobalContext &context) override;
    void Preprocess(ProcessingContext &context) override;
    void Process(ProcessingContext &context) override;
    void Postprocess(ProcessingContext &context) override;
    void Unprepare(GlobalContext &context) override;

  protected:
    // Data ports
    PortIn<MultiChannelType<double>>   *data_in_port_;
    PortOut<MultiChannelType<double>> *phase_out_port_;
    PortOut<MultiChannelType<double>> *real_out_port_;

    // Options
    options::Int n_messages_{-1};
    options::Bool calibrate_{false};
    options::Value<unsigned int, false> f0_read_interval_{5000};
    options::Value<YAML::Node, false> filter_def_{};
    options::Bool compensate_filter_{true};

    // Runtime state
    unsigned int packet_count_ = 0;
    double first_timestamp_ = 0.0;
    double fs_ = 0.0;
    double f0_ = 10.0;         // current f0 (updated from shared state)
    size_t n_fft_ = 0;
    size_t window_size_ = 1;

    boost::circular_buffer<double> sample_window{1}; // resized in Preprocess
    std::string coeff_file_;                         // path to bandpass filter coefficients
    std::vector<std::complex<double>> coeffs_;       // frequency-domain bandpass coefficients
    std::complex<double> c_gain_;                    // MSE-optimal calibration gain for cecHT

    std::vector<double> filter_phase_shift_values_; // phase shift per 0.1 Hz increment
    double filter_phase_shift_ = 0.0;               // active phase shift for current f0_

    FollowerState<double> *f0_state_ = nullptr;
    bool valid_f0_ = false;

    // FFTW resources (allocated in Preprocess, freed in Unprepare)
    double       *signal_in = nullptr;
    fftw_complex *freq_half = nullptr;
    fftw_complex *freq = nullptr;
    fftw_complex *out = nullptr;
    fftw_plan p_;
    fftw_plan p_inv_;

    const uint32_t MAX_NCHANNELS = 384;
};
