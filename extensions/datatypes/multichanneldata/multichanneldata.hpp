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

#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include "idata.hpp"
#include "utilities/general.hpp"
#include "utilities/iterators.hpp"
#include "utilities/string.hpp"

typedef Range<size_t> SampleRange;
template <typename T> class MultiChannelType;

namespace nsMultiChannel {

using Base = AnyType;

// Forward declaration
template <typename T> class Data;

struct Parameters : Base::Parameters {
  Parameters(size_t nchan = 0, size_t nsamp = 0, double rate = 1.0)
      : Base::Parameters(), nchannels(nchan), nsamples(nsamp),
        sample_rate(rate) {}

  size_t nchannels;
  size_t nsamples;
  double sample_rate;
};

class Capabilities : public Base::Capabilities {
 public:
  Capabilities(ChannelRange channel_range,
               SampleRange sample_range =
                   SampleRange(1, std::numeric_limits<uint32_t>::max()))
      : Base::Capabilities(), channel_range_(channel_range),
        sample_range_(sample_range) {}

  ChannelRange channel_range() const { return channel_range_; }
  SampleRange sample_range() const { return sample_range_; }

  virtual void VerifyCompatibility(const Capabilities &capabilities) const {
    if (!channel_range_.overlapping(capabilities.channel_range())) {
      throw std::runtime_error("Channel ranges do not overlap (" +
                               channel_range_.to_string() + " and " +
                               capabilities.channel_range().to_string() + ")");
    }
    if (!sample_range_.overlapping(capabilities.sample_range())) {
      throw std::runtime_error("Sample ranges do not overlap (" +
                               sample_range_.to_string() + " and " +
                               capabilities.sample_range().to_string() + ")");
    }
  }
  
  template<typename T>
  void Validate(const Data<T>& prototype) const {
    // Validate using the data object's properties
    Validate(Parameters(prototype.nchannels(), prototype.nsamples(), prototype.sample_rate()));
  }
  
  virtual void Validate(const Parameters &parameters) const {
    if (parameters.nsamples == 0 ||
        !sample_range_.inrange(parameters.nsamples)) {
      throw std::runtime_error(
          "Number of samples cannot be zero and needs to be in range " +
          sample_range_.to_string());
    }

    if (parameters.nchannels == 0 ||
        !channel_range_.inrange(parameters.nchannels)) {
      throw std::runtime_error(
          "Number of channels cannot be zero and needs to be in range " +
          channel_range_.to_string());
    }
  }

 protected:
  ChannelRange channel_range_;
  SampleRange sample_range_;
};

template <typename T> class Data : public Base::Data {
 public:
  typedef stride_iter<T *> channel_iterator;
  typedef T *sample_iterator;

  Data() {}

  Data(const Parameters& parameters) {
    Initialize(parameters);
  }

  Data(size_t nchannels, size_t nsamples, double sample_rate) {
    Initialize(nchannels, nsamples, sample_rate);
  }

  void ClearData() override {
    std::fill(data_.begin(), data_.end(), 0);
    std::fill(timestamps_.begin(), timestamps_.end(), 0);
  }

  void Initialize(const Parameters &parameters) {
    Initialize(parameters.nchannels, parameters.nsamples,
               parameters.sample_rate);
  }

  void Initialize(size_t nchannels, size_t nsamples, double sample_rate) {
    if (nchannels == 0 || nsamples == 0) {
      throw std::runtime_error(". MultiChannelData::Initialize - number of "
                               "channels/samples needs to be larger than 0.");
    }

    if (sample_rate <= 0) {
      throw std::runtime_error(". MultiChannelData::Initialize - sample rate "
                               "needs to be larger than 0.");
    }

    nchannels_ = nchannels;
    nsamples_ = nsamples;
    sample_rate_ = sample_rate;

    data_.resize(nchannels_ * nsamples_);

    timestamps_.resize(nsamples_);
  }

  size_t nchannels() const { return nchannels_; }
  size_t nsamples() const { return nsamples_; }
  double sample_rate() const { return sample_rate_; }

  uint64_t sample_timestamp(size_t sample = 0) const {
    return timestamps_[sample];
  }
  std::vector<uint64_t> &sample_timestamps() { return timestamps_; }

  void set_sample_timestamp(size_t sample, uint64_t t) {
    if (sample >= nsamples_) {
      throw std::out_of_range(". Sample index " + std::to_string(sample) +
                              " out of range. Max index is " +
                              std::to_string(nsamples_ - 1));
    } else {
      timestamps_[sample] = t;
    }
  }

  void set_sample_timestamps(std::vector<uint64_t> &t) {
    assert(t.size() == nsamples_);
    timestamps_ = t;
  }

  void set_data_channel(size_t channel, std::vector<T> &data) {
    assert(data.size() == nsamples_);
    T *ptr = data_.data();
    for (size_t k = 0; k < nsamples_; ++k) {
      (*ptr) = data[k];
      ptr += nchannels_;
    }
  }

  void set_data_sample(size_t sample, std::vector<T> &data) {
    assert(data.size() == nchannels_);
    std::copy(data.begin(), data.end(), begin_sample(sample));
  }

  void set_data_sample(size_t sample, size_t channel, T data) {
    if (sample >= nsamples_) {
      throw std::out_of_range(". Sample index " + std::to_string(sample) +
                              " out of range. Max index is " +
                              std::to_string(nsamples_ - 1));
    }
    if (channel >= nchannels_) {
      throw std::out_of_range(". Channel index " + std::to_string(sample) +
                              " out of range. Max index is " +
                              std::to_string(nchannels_ - 1));
    }
    data_[flat_index(sample, channel)] = data;
  }

  std::vector<T> &data() { return data_; }

  const T &data_sample(size_t sample, size_t channel = 0) const {
    return data_[flat_index(sample, channel)];
  }

  T &operator()(size_t index) { return data_[index]; }
  const T &operator()(size_t index) const { return data_[index]; }

  T &operator()(size_t sample, size_t channel = 0) {
    return data_[flat_index(sample, channel)];
  }
  const T &operator()(size_t sample, size_t channel = 0) const {
    return data_[flat_index(sample, channel)];
  }

  // iterators
  T *begin_sample(size_t sample) { return &data_[flat_index(sample)]; }
  T *end_sample(size_t sample) { return begin_sample(sample) + nchannels_; }
  const T *begin_sample(size_t sample) const {
    return &data_[flat_index(sample)];
  }
  const T *end_sample(size_t sample) const {
    return begin_sample(sample) + nchannels_;
  }

  stride_iter<T *> begin_channel(size_t channel) {
    return stride_iter<T *>(&data_[channel], nchannels_);
  }
  stride_iter<T *> end_channel(size_t channel) {
    return begin_channel(channel) + nsamples_;
  }

  void SerializeBinary(std::ostream &stream,
                       Serialization::Format format =
                                   Serialization::Format::FULL) const override {
    AnyType::Data::SerializeBinary(stream, format);
    if (format == Serialization::Format::FULL) {
      stream.write(reinterpret_cast<const char *>(timestamps_.data()),
                   timestamps_.size() * sizeof(uint64_t));
      stream.write(reinterpret_cast<const char *>(data_.data()),
                   data_.size() * sizeof(T));
    }

    if (format == Serialization::Format::COMPACT) {
      for (size_t k = 0; k < nsamples_; ++k) {
        stream.write(reinterpret_cast<const char *>(&timestamps_[k]),
                     sizeof(uint64_t));
        stream.write(reinterpret_cast<const char *>(&data_[flat_index(k)]),
                     sizeof(T) * nchannels_);
      }
    }
  }

  void SerializeYAML(YAML::Node &node,
                     Serialization::Format format =
                                 Serialization::Format::FULL) const override {
    AnyType::Data::SerializeYAML(node, format);
    if (format == Serialization::Format::FULL ||
        format == Serialization::Format::COMPACT) {
      node["timestamps"] = timestamps_;
      // TODO: write samples individually to list of lists, instead of a single
      // flat list
      node["signal"] = data_;
    }
  }

  void SerializeFlatBuffer(flexbuffers::Builder& flex_builder) override{
      AnyType::Data::SerializeFlatBuffer(flex_builder);
      flex_builder.TypedVector("data", [&]{
             for(auto samples: data_)
                 flex_builder.Add(samples);
      });

      flex_builder.TypedVector("timestamps", [&]{
             for(auto samples: timestamps_)
                 flex_builder.Add(samples);
      });

      flex_builder.UInt("nchannels", nchannels());
      flex_builder.UInt("nsamples", nsamples());
      flex_builder.String("type", MultiChannelType<T>::datatype());
  }

  void YAMLDescription(YAML::Node &node,
                       Serialization::Format format =
                                   Serialization::Format::FULL) const override {
    AnyType::Data::YAMLDescription(node, format);
    if (format == Serialization::Format::FULL) {
      node.push_back("timestamps uint64 (" + std::to_string(nsamples_) + ")");
      node.push_back("signal " + get_type_string<T>() + " (" +
                     std::to_string(nchannels_) + "," +
                     std::to_string(nsamples_) + ")");
    }

    if (format == Serialization::Format::COMPACT) {
      node.push_back("timestamps uint64 (1)");
      node.push_back("signal " + get_type_string<T>() + " (" +
                     std::to_string(nchannels_) + ")");
    }
  }

  T sum_abs_sample(size_t sample) const {
    return std::accumulate(begin_sample(sample), end_sample(sample), 0,
                           [](T a, T b) { return a + std::abs(b); });
  }

  T sum_sample(size_t sample) const {
    return std::accumulate(begin_sample(sample), end_sample(sample), 0.0);
  }

  T mean_abs_sample(size_t sample) const {
    return sum_abs_sample(sample) / nchannels_;
  }

  T mean_sample(size_t sample) const { return sum_sample(sample) / nchannels_; }

 protected:
  inline size_t flat_index(size_t sample, size_t channel) const {
    return channel + sample * nchannels_;
  }
  inline size_t flat_index(size_t sample) const { return sample * nchannels_; }

 protected:
  size_t nchannels_;
  size_t nsamples_;
  double sample_rate_;
  std::vector<T> data_;
  std::vector<uint64_t> timestamps_;
};

}  // namespace nsMultiChannel

template <typename T> class MultiChannelType {
 public:
  static const std::string datatype() { return "multichannel"; }
  static const std::string dataname() { return "data"; }

  using Base = nsMultiChannel::Base;
  using Parameters = nsMultiChannel::Parameters;
  using Capabilities = nsMultiChannel::Capabilities;
  using Data = nsMultiChannel::Data<T>;
};
