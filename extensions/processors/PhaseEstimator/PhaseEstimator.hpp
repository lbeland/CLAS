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
#include <boost/circular_buffer.hpp>
#include <fftw3.h>

class PhaseEstimator : public IProcessor {
 public:
    PhaseEstimator();
    int get_max_bin(fftwf_complex* out, size_t out_size);
    void construct_analytic_spectrum(int M, const fftwf_complex* half, fftwf_complex* full);
    void fftshift(const fftwf_complex* in, fftwf_complex* out, int L);
    void ifftshift(const fftwf_complex* in, fftwf_complex* out, int L);

  void CreatePorts() override;
  void CompleteStreamInfo() override;
  void Preprocess(ProcessingContext &context) override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;


 protected:
  PortIn<MultiChannelType<float>> *data_in_port_;
  PortOut<MultiChannelType<float>> *data_out_port_;

  options::Value<unsigned int, false> n_messages_{0};
  options::Value<unsigned int, false> n_fft_{4096};
  
  unsigned int packet_count_ = 0;
  double first_timestamp_ = 0.0;
  float fs_ = 0.0;
  float f0_ = 0.0;
  inline static boost::circular_buffer<float> sample_window{1};  // Initialized with size 1, will be resized in Preprocess
};
