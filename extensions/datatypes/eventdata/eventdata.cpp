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

#include <string>

#include "eventdata.hpp"

using namespace nsEventType;

Data::Data(std::string event) { set_event(event); }

void Data::Initialize(std::string event) { set_event(event); }

void Data::ClearData() { set_event(DEFAULT_EVENT); }

std::string Data::event() const { return event_; }

size_t Data::hash() const { return hash_; }

size_t Data::size() const { return event_.size(); }

void Data::set_event(std::string event) {
  event_ = event;
  hash_ = std::hash<std::string>()(event_);
}

void Data::set_event(const Data &source) {
  event_ = source.event();
  hash_ = source.hash();
}

namespace nsEventType {

bool operator==(const Data &e1, const Data &e2) { return e1.hash_ == e2.hash_; }

bool operator!=(const Data &e1, const Data &e2) { return e1.hash_ != e2.hash_; }
}  // namespace nsEventType

void Data::SerializeBinary(std::ostream &stream,
                           Serialization::Format format) const {
  Base::Data::SerializeBinary(stream, format);
  if (format == Serialization::Format::FULL ||
      format == Serialization::Format::COMPACT) {
    std::string buffer = event_;
    buffer.resize(EVENT_STRING_LENGTH);
    stream.write(buffer.data(), EVENT_STRING_LENGTH);
  }
}

void Data::SerializeYAML(YAML::Node &node, Serialization::Format format) const {
  Base::Data::SerializeYAML(node, format);
  if (format == Serialization::Format::FULL ||
      format == Serialization::Format::COMPACT) {
    node["event"] = event_;
  }
}

void Data::SerializeFlatBuffer(flexbuffers::Builder& flex_builder){
    Base::Data::SerializeFlatBuffer(flex_builder);
    flex_builder.String("event", event_);
    flex_builder.String("type", EventType::datatype());
}

void Data::YAMLDescription(YAML::Node &node,
                           Serialization::Format format) const {
  Base::Data::YAMLDescription(node, format);
  if (format == Serialization::Format::FULL ||
      format == Serialization::Format::COMPACT) {
    node.push_back("event_string str (" + std::to_string(EVENT_STRING_LENGTH) +
                   ")");
  }
}
