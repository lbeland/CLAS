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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "iprocessor.hpp"
#include "multichanneldata/multichanneldata.hpp"
#include "options/options.hpp"
#include "utilities/time.hpp"

// ---------------------------------------------------------------------------
// Field descriptor parsed from the YAML header's "data" list.
// Each entry encodes name, element type, and shape, e.g.
//   "signal float32 (32,64)"
//   "hardware_ts uint64 (1)"
// ---------------------------------------------------------------------------
struct FieldDescriptor {
  std::string name;
  std::string dtype;     // "float32", "uint64", ...
  std::vector<int> dims; // shape, e.g. {32, 64} or {1}
  std::size_t n_items;   // product of dims
  std::size_t byte_size; // n_items * sizeof(element)
  std::size_t offset;    // byte offset within one record
};

class ReplaySource : public IProcessor {
  public:
    ReplaySource();

    void CreatePorts() override;
    void CompleteStreamInfo() override;
    void Prepare(GlobalContext &context) override;
    void Preprocess(ProcessingContext &context) override;
    void Process(ProcessingContext &context) override;
    void Postprocess(ProcessingContext &context) override;

  protected:
    // Data ports
    PortOut<MultiChannelType<double>> *data_out_port_;

    // Options - replay source selection
    options::String path_{"run://"};
    options::String file_{""};
    options::Int    slot_{0};

    // Options - playback controls
    options::Bool   loop_{false};
    options::Bool   real_time_{true};
    options::Double speed_factor_{1.0};
    options::Int    n_messages_{-1};

    // Options - output contract
    options::Value<unsigned int, false> nchannels_{32};
    options::Value<unsigned int, false> nsamples_{1};
    options::Double fs_{1000.0};

  private:
    // Helpers
    std::string ResolveFilePath(const std::string path) const;
    void        LoadFile(const std::string &filepath);
    static FieldDescriptor ParseDataEntry(const std::string &entry);
    static std::size_t     DtypeSize(const std::string &dtype);

    // Parsed file state
    std::vector<FieldDescriptor> layout_;   // field descriptors in record order
    std::size_t record_size_{0};            // total bytes per record
    std::size_t n_records_{0};              // number of complete records in the file
    std::size_t signal_offset_{0};          // byte offset of the "signal" field
    std::string signal_dtype_{"float32"};   // on-disk element type of the "signal" field
    std::vector<std::uint8_t> payload_;     // raw binary payload (after YAML header)

    // Runtime replay state
    std::size_t current_record_{0};         // index of the next record to emit
    TimePoint   last_emit_time_{};          // wall-clock time of the previous emit
    double      inter_packet_ns_{0.0};      // expected gap between consecutive packets [ns]
    std::int64_t emitted_{0};               // total packets emitted in the current run
};
