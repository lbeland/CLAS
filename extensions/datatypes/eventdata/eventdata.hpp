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

#include <string>

#include "idata.hpp"
#include "yaml-cpp/yaml.h"

typedef unsigned int EventIDType;

const std::string DEFAULT_EVENT = "none";
// to be used for port names using event data
const std::string EVENTDATA = "events";

namespace nsEventType {

using Base = AnyType;

struct Parameters : Base::Parameters {
  Parameters(std::string event = DEFAULT_EVENT)
      : Base::Parameters(), default_event(event) {}

  std::string default_event;
};

class Capabilities : public Base::Capabilities {
 public:
  virtual void Validate(const Parameters &parameters) const {
    if (parameters.default_event.size() == 0) {
      throw std::runtime_error("Default event string cannot be empty.");
    }
  }
};

class Data : public Base::Data {
 public:
  Data(std::string event = DEFAULT_EVENT);

  void Initialize(std::string event = DEFAULT_EVENT);
  void Initialize(const Parameters &parameters) {
    set_event(parameters.default_event);
  }

  void ClearData() override;
  std::string event() const;
  size_t hash() const;
  size_t size() const;
  void set_event(std::string event);
  void set_event(const Data &source);

  friend bool operator==(const Data &e1, const Data &e2);
  friend bool operator!=(const Data &e1, const Data &e2);

  void SerializeBinary(std::ostream &stream,
                       Serialization::Format format =
                                   Serialization::Format::FULL) const override;
  void SerializeYAML(YAML::Node &node,
                     Serialization::Format format =
                                 Serialization::Format::FULL) const override;

  void SerializeFlatBuffer(flexbuffers::Builder& fbb) override;

  void YAMLDescription(YAML::Node &node,
                       Serialization::Format format =
                                   Serialization::Format::FULL) const override;

 protected:
  std::string event_;
  size_t hash_;

  static const unsigned int EVENT_STRING_LENGTH = 128;
};

}  // namespace nsEventType

class EventType {
 public:
  static const std::string datatype() { return "event"; }
  static const std::string dataname() { return "events"; }

  using Base = nsEventType::Base;
  using Parameters = nsEventType::Parameters;
  using Capabilities = nsEventType::Capabilities;
  using Data = nsEventType::Data;
};

namespace YAML {
template <> struct convert<EventType::Data> {
  static Node encode(const EventType::Data &rhs) {
    Node node;
    node = rhs.event();
    return node;
  }

  static bool decode(const Node &node, EventType::Data &rhs) {
    rhs.set_event(node.as<std::string>());
    return true;
  }
};
}  // namespace YAML
