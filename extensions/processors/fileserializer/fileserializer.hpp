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
#include <memory>
#include <vector>

#include "iprocessor.hpp"
#include "options/options.hpp"
#include "serializer.hpp"

class FileSerializer : public IProcessor {
  // CONSTRUCTOR and OVERLOADED METHODS
 public:
  FileSerializer();
  void CreatePorts() override;
  void Configure(const GlobalContext &context) override;
  void Preprocess(ProcessingContext &context) override;
  void Process(ProcessingContext &context) override;
  void Postprocess(ProcessingContext &context) override;

  // METHODS
 protected:
  /* Add YAML preamble to file (note: there is one file by slot in the input
   * port)
   *
   * @input out file stream to emit the preamble node
   * @input slot the slot number corresponding to the file
   */
  void create_preamble(std::ostream &out, int slot);

  // DATA PORTS
 protected:
  PortIn<AnyType> *data_port_;

  // VARIABLES
 protected:
  std::unique_ptr<Serialization::Serializer> serializer_;
  std::vector<std::unique_ptr<std::ostream>> streams_;
  std::vector<uint64_t> packetid_;
  std::vector<unsigned int> upstream_buffer_size_;
  double throttle_level_;
  std::vector<uint64_t> nskipped_;

  // OPTIONS
 protected:
  options::String path_{"run://"};
  options::Value<Serialization::Encoding, false> encoding_{
      Serialization::Encoding::BINARY};
  options::Value<Serialization::Format, false> format_{
      Serialization::Format::FULL};
  options::Bool overwrite_{false};
  options::Bool throttle_{false};
  options::Double throttle_threshold_{0.3, options::inrange<double>(0., 1.)};
  options::Double throttle_smooth_{0.5, options::inrange<double>(0., 1.)};
  options::Bool preamble_{true};
};
