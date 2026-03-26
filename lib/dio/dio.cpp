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

#include "dio.hpp"
#include "utilities/time.hpp"

uint32_t DigitalState::nchannels() const { return state_.size(); }

std::vector<bool> &DigitalState::state() { return state_; }

bool DigitalState::state(uint32_t channel) const {
  if (channel >= nchannels()) {
    throw DigitalStateError("Invalid channel number.");
  }
  return state_[channel];
}

typename std::vector<bool>::reference
DigitalState::operator[](uint32_t channel) {
  return state_[channel];
}

std::vector<bool> DigitalState::state(std::vector<uint32_t> channels) const {
  std::vector<bool> ret;

  for (auto &it : channels) {
    if (it >= nchannels()) {
      throw DigitalStateError("Invalid channel number");
    }
    ret.push_back(state_[it]);
  }
  return ret;
}

void DigitalState::set_state(uint32_t channel, bool value) {
  if (channel >= nchannels()) {
    throw DigitalStateError("Invalid channel number.");
  }
  state_[channel] = value;
}

void DigitalState::set_state(std::vector<uint32_t> channels, bool value) {
  for (auto &it : channels) {
    if (it >= nchannels()) {
      continue;
    }  // fail silently
    state_[it] = value;
  }
}

void DigitalState::set_state(bool value) { state_.assign(nchannels(), value); }

void DigitalState::set_state(std::vector<bool> values) {
  if (values.size() != nchannels()) {
    throw DigitalStateError("Incorrect size of vector.");
  }
  state_ = values;
}

void DigitalState::set_state(std::vector<uint32_t> channels,
                             std::vector<bool> values) {
  if (channels.size() != values.size()) {
    throw DigitalStateError("Incompatible size of vectors.");
  }

  for (unsigned int k = 0; k < channels.size(); ++k) {
    if (channels[k] >= nchannels()) {
      continue;
    }  // fail silently
    state_[channels[k]] = values[k];
  }
}

void DigitalState::toggle_state(uint32_t channel) {
  set_state(channel, !state(channel));
}

void DigitalState::toggle_state(std::vector<uint32_t> channels) {
  for (auto &it : channels) {
    if (it >= nchannels()) {
      continue;
    }  // fail silently
    state_[it] = !state_[it];
  }
}

std::string DigitalState::to_string(std::string high, std::string low,
                                    std::string spacer) const {
  std::string s = "";
  for (uint32_t k = 0; k < nchannels(); ++k) {
    if (k > 0) {
      s += spacer;
    }
    s += state_[k] ? high : low;
  }
  return s;
}

std::string DigitalDevice::type() const { return type_; }

std::string DigitalDevice::description() const {
  return type() + " (" + std::to_string(nchannels()) + " channels)";
}

DigitalOutputProtocol::DigitalOutputProtocol(uint32_t nchannels,
                                             unsigned int pulse_width,
                                             DigitalOutputMode default_mode)
    : nchannels_(nchannels), pulse_width_(pulse_width) {
  mode_.assign(nchannels_, default_mode);
}

void DigitalOutputProtocol::set_mode(uint32_t channel, DigitalOutputMode mode) {
  if (channel <= nchannels_) {  // fail silently
    mode_[channel] = mode;
  }
}

unsigned int DigitalOutputProtocol::pulse_width() const { return pulse_width_; }

void DigitalOutputProtocol::set_pulse_width(unsigned int value) {
  pulse_width_ = value;
}

void DigitalOutputProtocol::set_mode(std::vector<uint32_t> channels,
                                     DigitalOutputMode mode) {
  for (const uint32_t &c : channels) {
    if (c <= nchannels_) {
      mode_[c] = mode;
    }
  }
}

std::vector<uint32_t>
DigitalOutputProtocol::find_channels(DigitalOutputMode mode) {
  std::vector<uint32_t> channels;
  for (uint32_t k = 0; k < mode_.size(); ++k) {
    if (mode_[k] == mode) {
      channels.push_back(k);
    }
  }
  return channels;
}

void DigitalOutputProtocol::execute(DigitalDevice &device) {
  DigitalState state(nchannels_);
  std::vector<uint32_t> channels;
  state = device.read_state();
  channels = find_channels(DigitalOutputMode::HIGH);
  state.set_state(channels, true);
  channels = find_channels(DigitalOutputMode::LOW);
  state.set_state(channels, false);
  channels = find_channels(DigitalOutputMode::TOGGLE);
  state.toggle_state(channels);
  channels = find_channels(DigitalOutputMode::PULSE);
  state.set_state(channels, true);
  device.write_state(state);

  if (channels.size() > 0) {  // some channels are pulsed
    custom_sleep_for(pulse_width_);
    state.set_state(channels, false);
    device.write_state(state);
  }
}
