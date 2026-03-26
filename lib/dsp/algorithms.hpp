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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dsp {
namespace algorithms {

enum class Slope { UP, DOWN };

class ThresholdCrosser {
 public:
  ThresholdCrosser(double threshold, Slope slope = Slope::UP)
      : threshold_(threshold), slope_(slope) {}

  double threshold() const;
  void set_threshold(double value);

  Slope slope() const;
  void set_slope(Slope value);

  bool has_crossed(double sample);
  bool has_crossed_up(double sample);
  bool has_crossed_down(double sample);

 private:
  double threshold_;
  Slope slope_;
  double prev_sample_;
};

class RunningStatistics {
 public:
  RunningStatistics(double alpha, uint64_t burn_in = 0,
                    bool outlier_protection = false, double outlier_zscore = 3,
                    double outlier_half_life = 1, double center = 0.0,
                    double dispersion = 0.0);

  double alpha() const;
  uint64_t burn_in() const;
  double center() const;
  double dispersion() const;

  bool outlier_protection() const;
  double outlier_zscore() const;
  double outlier_half_life() const;

  bool is_burning_in() const;

  double zscore(double value) const;

  void set_center(double value);
  void set_dispersion(double value);
  void set_alpha(double value);
  void set_burn_in(uint64_t value);

  void set_outlier_protection(bool value);
  void set_outlier_zscore(double value);
  void set_outlier_half_life(double value);

  void add_sample(double sample);
  void add_samples(std::vector<double> samples);

  template <typename Iter> void add_samples(Iter begin, Iter end) {
    for (; begin != end; ++begin) {
      add_sample(*begin);
    }
  }

 protected:
  double center_;
  double dispersion_;

 protected:
  virtual void update_statistics(double sample, double alpha) = 0;

 private:
  double alpha_;
  uint64_t burn_in_;
  uint64_t burn_in_counter_;

  bool outlier_protection_;
  double outlier_zscore_;
  double outlier_half_life_;
};

class RunningMeanMAD : public RunningStatistics {
 public:
  RunningMeanMAD(double alpha = 1.0, unsigned int burn_in = 0,
                 bool outlier_protection = false, double outlier_zscore = 3,
                 double outlier_half_life = 1, double mean = 0.0,
                 double mad = 0.0)
      : RunningStatistics(alpha, burn_in, outlier_protection, outlier_zscore,
                          outlier_half_life, mean, mad) {}

  double mean() const;
  double mad() const;

 protected:
  virtual void update_statistics(double sample, double alpha);
};

class PeakDetector {
 public:
  PeakDetector(uint64_t init_timestamp = 0, double init_value = 0.0)
      : last_slope_is_up_(false), previous_value_(init_value),
        previous_timestamp_(init_timestamp), npeaks_found_(0),
        last_peak_amplitude_(0), last_peak_timestamp_(0) {}

  void reset(uint64_t init_timestamp = 0, double init_value = 0.0);

  double last_peak_amplitude() const;
  uint64_t last_peak_timestamp() const;

  bool is_peak(const uint64_t &timestamp, const double &sample);

  bool upslope() const;

  uint64_t npeaks() const;

 private:
  bool last_slope_is_up_;

  double previous_value_;
  uint64_t previous_timestamp_;

  uint64_t npeaks_found_;

  double last_peak_amplitude_;
  uint64_t last_peak_timestamp_;
};

class ExponentialSmoother {
 public:
  ExponentialSmoother(double alpha, double init_value = 0.0);

  double smooth(double value);

  double alpha() const;
  void set_alpha(double value);

  double value() const;
  void set_value(double value);

 private:
  double alpha_;
  double value_;
};

enum class SpikeDetectionMode { PEAK = 0, THRESHOLD };

class SpikeDetector {
 public:
  SpikeDetector(unsigned int nchannels, double threshold,
                unsigned int peak_life_time);

  ~SpikeDetector() {}

  void reset();

  unsigned int nchannels() const;

  double threshold() const;
  void set_threshold(double value);

  unsigned int peak_life_time() const;
  void set_peak_life_time(unsigned int value);

  uint64_t timestamp_detected_spike() const;
  const std::vector<double> &amplitudes_detected_spike() const;

  /**
   * Spike detection algorithm:
   *
   * - operates sample by sample
   * - looks for upwards deflections in at least one of the channels above
   * a certain threshold
   * - a spike is detected if the signal of at least one channel crosses the
   * threshold and a local maxima is found in at least one channel (not
   * necessarily the same of that of threshold crossing) within a certain
   * duration (determined by the peak lifetime)
   * - the timestamp of the detected spike corresponds to that one of the first
   * sample that crossed the threshold first (independently on whether that
   * sample belongs to the current or previous buffers)
   * - in case a proper maximum is found on all channels, the peak values are
   * returned, together with the threshold-crossing timestamp; however, if on
   * one or more channels no peaks were found, the values of the signals at the
   * threshold-crossing sample will be returned.
   */
  template <typename ForwardIterator>
  bool is_spike(const uint64_t timestamp, const ForwardIterator sample) {
    unsigned int c;
    auto spike_found = false;
    auto it = sample;

    if (detection_mode_ == SpikeDetectionMode::THRESHOLD) {
      // is threshold crossed on any of the channels?
      for (c = 0; c < nchannels_; ++c) {
        if (previous_sample_[c] <= threshold_ && *it > threshold_) {
          // std::cout << ". Threshold crossed at timestamp: " << timestamp <<
          // std::endl;
          detection_mode_ = SpikeDetectionMode::PEAK;
          prepare_peak_detection(timestamp, sample);
          break;
        }
        ++it;
      }

    } else if (detection_mode_ == SpikeDetectionMode::PEAK) {
      // look for peaks
      for (c = 0; c < nchannels_; ++c) {
        if (!peak_found_[c]) {
          if (slope_[c] > 0 && *it < previous_sample_[c]) {
            peak_found_[c] = true;
            ++npeaks_found_;
            peak_amplitudes_[c] = previous_sample_[c];
          }
        }
        ++it;
      }

      --peak_countdown_;

      if (peak_countdown_ == 0 || npeaks_found_ == nchannels_) {
        // QUESTION: should we wait until peak_life_time window is over, or
        // restart threshold detection as soon as a peak was found on all
        // channels

        // set spike amplitude of channels without peak
        // QUESTION: should we only accepts spikes with peaks in all channels?
        // QUESTION: if not, what amplitude should we assign to the channels
        // without peak (e.g. some invalid value, or current sample)
        if (npeaks_found_ > 0) {  // spike found!!
          ++nspikes_found_;
          spike_found = true;
        }

        detection_mode_ = SpikeDetectionMode::THRESHOLD;

      } else {
        update_slope(sample);
      }
    }

    std::copy_n(sample, nchannels_, previous_sample_.begin());

    return spike_found;
  }

  bool is_spike(const uint64_t timestamp, const std::vector<double> &sample);

  uint64_t nspikes() const;

 private:
  template <typename ForwardIterator>
  void update_slope(ForwardIterator sample) {
    for (decltype(nchannels_) channel = 0; channel < nchannels_; ++channel) {
      if (previous_sample_[channel] != *sample) {  // deal with plateaus
        slope_[channel] = *sample - previous_sample_[channel];
      }
      ++sample;
    }
  }

  template <typename ForwardIterator>
  void prepare_peak_detection(const uint64_t timestamp,
                              const ForwardIterator sample) {
    spike_timestamp_ = timestamp;

    peak_countdown_ = peak_life_time_;
    npeaks_found_ = 0;
    peak_found_.assign(nchannels_, false);
    peak_amplitudes_ = previous_sample_;
    // if no peak found, we will return the detection sample

    update_slope(sample);
  }

 private:
  unsigned int nchannels_;
  double threshold_;
  unsigned int peak_life_time_;
  uint64_t nspikes_found_;

 private:
  SpikeDetectionMode detection_mode_;
  std::vector<double> previous_sample_;
  uint64_t spike_timestamp_;
  std::vector<double> slope_;
  unsigned int peak_countdown_;
  std::vector<bool> peak_found_;
  unsigned int npeaks_found_;
  std::vector<double> peak_amplitudes_;
};

}  // namespace algorithms
}  // namespace dsp
