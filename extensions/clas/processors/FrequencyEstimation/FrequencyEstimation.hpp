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
#include <dsp/gram_savitzky_golay.hpp>
#include <dsp/filter.hpp>
#include <boost/circular_buffer.hpp>
#include <complex>
#include <fftw3.h>
#include <vector>

class FrequencyEstimation : public IProcessor {
  public:
    FrequencyEstimation();

    void CreatePorts() override;
    void CompleteStreamInfo() override;
    void Prepare(GlobalContext &context) override;
    void Preprocess(ProcessingContext &context) override;
    void Process(ProcessingContext &context) override;
    void Postprocess(ProcessingContext &context) override;
    void Unprepare(GlobalContext &context) override;

  protected:
    // Data ports
    PortIn<MultiChannelType<double>>  *data_in_port_;
    PortOut<ScalarType<double>>      *data_out_port_;

    // Options
    options::Int    n_messages_{-1};
    options::Double window_size_sec_{5};
    options::Double f_min_{5.0};
    options::Double f_max_{18.0};
    options::Value<unsigned int, false> calc_interval_{100};
    options::Double max_invalid_sec_{1.0};
    options::Double kalman_f0_std_{0.397857};
    options::Bool   kalman_full_{true};
    options::Double max_gauss_width_hz_{2.0};
    options::Bool   debug_output_{false};

    // Runtime state - shared f0 broadcaster
    BroadcasterState<double> *f0_state_ = nullptr;
    double current_f0_ = std::numeric_limits<double>::quiet_NaN();
    double last_valid_f0_ = std::numeric_limits<double>::quiet_NaN();
    double current_gauss_sigma_ = std::numeric_limits<double>::quiet_NaN();

    // Kalman filter state
    double kf_x_ = 0.0;  // state estimate (smoothed f0)
    double kf_P_ = 0.0;  // state uncertainty
    double kf_Q_ = 0.0;  // process noise variance per update step
    double kf_R_ = 0.0;  // measurement noise variance

    // Signal processing parameters
    unsigned int packet_count_ = 0;
    double freq_resolution_ = 0.0;
    double fs_ = 0.0;
    double SNR_ = 1.0;
    size_t n_fft_ = 0;
    size_t window_size_ = 1;
    int max_analyze_bin_ = 0;
    int f_min_bin_ = 0;
    int f_max_bin_ = 0;
    int invalid_count_ = 0;
    int invalid_threshold_ = 0;
    bool debug_out_active_ = false; // resolved in Prepare: debug_output_ && data-out port has a slot 1

    boost::circular_buffer<double> sample_window{1}; // resized in Prepare
    std::vector<double> savgol_weights_; // Savitzky-Golay smoothing weights, evaluated at t=0 (see Prepare)

    // FFTW resources (allocated in Prepare, freed in Unprepare)
    fftw_plan    fft_plan_ = nullptr;
    double       *signal_in = nullptr;
    fftw_complex *freq_half = nullptr;
    std::vector<double> freqs_;

    const uint32_t MAX_NCHANNELS = 384;
};
