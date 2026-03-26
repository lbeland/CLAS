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

#include "algorithms.hpp"

using namespace dsp::algorithms;

double ThresholdCrosser::threshold() const { return threshold_; }

void ThresholdCrosser::set_threshold(double value) { threshold_ = value; }

Slope ThresholdCrosser::slope() const { return slope_; }

void ThresholdCrosser::set_slope(Slope value) { slope_ = value; }

bool ThresholdCrosser::has_crossed(double sample) {
  if (slope_ == Slope::UP) {
    return has_crossed_up(sample);
  } else {
    return has_crossed_down(sample);
  }
}

bool ThresholdCrosser::has_crossed_up(double sample) {
  bool ret = prev_sample_ <= threshold_ && sample > threshold_;
  prev_sample_ = sample;
  return ret;
}

bool ThresholdCrosser::has_crossed_down(double sample) {
  bool ret = prev_sample_ >= threshold_ && sample < threshold_;
  prev_sample_ = sample;
  return ret;
}

RunningStatistics::RunningStatistics(double alpha, uint64_t burn_in,
                                     bool outlier_protection,
                                     double outlier_zscore,
                                     double outlier_half_life, double center,
                                     double dispersion)
    : outlier_protection_(outlier_protection) {

  set_alpha(alpha);
  burn_in_counter_ = burn_in;
  set_burn_in(burn_in);

  set_outlier_zscore(outlier_zscore);
  set_outlier_half_life(outlier_half_life);

  set_center(center);
  set_dispersion(dispersion);


}

double RunningStatistics::alpha() const { return alpha_; }

uint64_t RunningStatistics::burn_in() const { return burn_in_; }

double RunningStatistics::center() const { return center_; }

double RunningStatistics::dispersion() const { return dispersion_; }

bool RunningStatistics::outlier_protection() const {
  return outlier_protection_;
}

double RunningStatistics::outlier_zscore() const { return outlier_zscore_; }

double RunningStatistics::outlier_half_life() const {
  return outlier_half_life_;
}

bool RunningStatistics::is_burning_in() const { return burn_in_counter_ > 0; }

double RunningStatistics::zscore(double value) const {
  return (value - center_) / dispersion_;
}

void RunningStatistics::set_outlier_protection(bool value) {
  outlier_protection_ = value;
}

void RunningStatistics::set_outlier_zscore(double value) {
  if (value <= 0) {
    throw std::out_of_range("Outlier zscore should be larger than zero.");
  }
  outlier_zscore_ = value;
}

void RunningStatistics::set_outlier_half_life(double value) {
  if (value <= 0) {
    throw std::out_of_range("Outlier half life should be larger than zero.");
  }
  outlier_half_life_ = value;
}

void RunningStatistics::set_center(double value) { center_ = value; }

void RunningStatistics::set_dispersion(double value) {  if (value < 0) {
    throw std::out_of_range("Dispersion should be equal to or larger than 0.");
  }
  dispersion_ = value;
}

void RunningStatistics::set_alpha(double value) {
  if (value < 0 || value > 1) {
    throw std::out_of_range("Alpha should be in range 0-1.");
  }
  alpha_ = value;
}

void RunningStatistics::set_burn_in(uint64_t value) {
  burn_in_ = value;
  if (burn_in_counter_ > burn_in_) {
    burn_in_counter_ = burn_in_;
  }
}

void RunningStatistics::add_sample(double sample) {
  double alpha = alpha_;

  // adjust alpha during burn-in or for outliers
  if (burn_in_counter_ > 0) {
    --burn_in_counter_;
    alpha = alpha + (1.0 - alpha) / (burn_in_ - burn_in_counter_);
  } else if (outlier_protection_) {
    double z = std::abs(zscore(sample));
    if (z > outlier_zscore_) {
      alpha = alpha * std::pow(2, (outlier_zscore_ - z) / outlier_half_life_);
    }
  }

  update_statistics(sample, alpha);
}

void RunningStatistics::add_samples(std::vector<double> samples) {
  for (auto &it : samples) {
    add_sample(it);
  }
}

double RunningMeanMAD::mean() const { return center_; }

double RunningMeanMAD::mad() const { return dispersion_; }

void RunningMeanMAD::update_statistics(double sample, double alpha) {
  center_ = (1 - alpha) * center_ + alpha * sample;
  dispersion_ = (1 - alpha) * dispersion_ + alpha * std::abs(sample - center_);
}

void PeakDetector::reset(uint64_t init_timestamp, double init_value) {
  previous_value_ = init_value;
  previous_timestamp_ = init_timestamp;
  last_slope_is_up_ = false;

  npeaks_found_ = 0;
  last_peak_timestamp_ = 0;
  last_peak_amplitude_ = 0.0;
}

bool PeakDetector::is_peak(const uint64_t &timestamp, const double &sample) {
  double diff = sample - previous_value_;
  bool peak = diff < 0 && last_slope_is_up_;

  if (peak) {
    ++npeaks_found_;
    last_peak_amplitude_ = previous_value_;
    last_peak_timestamp_ = previous_timestamp_;
  }

  previous_value_ = sample;
  previous_timestamp_ = timestamp;

  if (diff != 0) {
    last_slope_is_up_ = diff > 0;
  }

  return peak;
}

double PeakDetector::last_peak_amplitude() const {
  return last_peak_amplitude_;
}

uint64_t PeakDetector::last_peak_timestamp() const {
  return last_peak_timestamp_;
}

bool PeakDetector::upslope() const { return last_slope_is_up_; }

uint64_t PeakDetector::npeaks() const { return npeaks_found_; }

ExponentialSmoother::ExponentialSmoother(double alpha, double init_value)
    : value_(init_value) {

  set_alpha(alpha);
}

void ExponentialSmoother::set_alpha(double value) {
  if (value < 0 || value > 1) {
    throw std::out_of_range("Alpha should be in range 0-1.");
  }
  alpha_ = value;
}

double ExponentialSmoother::smooth(double value) {
  value_ = alpha_ * value + (1 - alpha_) * value_;
  return value_;
}

double ExponentialSmoother::alpha() const { return alpha_; }

double ExponentialSmoother::value() const { return value_; }

void ExponentialSmoother::set_value(double value) { value_ = value; }

SpikeDetector::SpikeDetector(unsigned int nchannels, double threshold,
                             unsigned int peak_life_time)
    : nchannels_(nchannels), threshold_(threshold),
      peak_life_time_(peak_life_time) {

  reset();
}

bool SpikeDetector::is_spike(const uint64_t timestamp,
                             const std::vector<double> &sample) {
  return is_spike(timestamp, sample.begin());
}

unsigned int SpikeDetector::nchannels() const { return nchannels_; }

double SpikeDetector::threshold() const { return threshold_; }

void SpikeDetector::set_threshold(double value) { threshold_ = value; }

unsigned int SpikeDetector::peak_life_time() const { return peak_life_time_; }

void SpikeDetector::set_peak_life_time(unsigned int value) {
  peak_life_time_ = value;
}

uint64_t SpikeDetector::timestamp_detected_spike() const {
  return spike_timestamp_;
}

const std::vector<double> &SpikeDetector::amplitudes_detected_spike() const {
  return peak_amplitudes_;
}

uint64_t SpikeDetector::nspikes() const { return nspikes_found_; }

void SpikeDetector::reset() {
  previous_sample_.assign(nchannels_, 0);
  peak_countdown_ = 0;
  slope_.assign(nchannels_, 0.0);
  spike_timestamp_ = 0;

  nspikes_found_ = 0;

  peak_found_.assign(nchannels_, false);
  peak_amplitudes_.assign(nchannels_, 0.0);
  npeaks_found_ = 0;

  detection_mode_ = SpikeDetectionMode::THRESHOLD;
}
