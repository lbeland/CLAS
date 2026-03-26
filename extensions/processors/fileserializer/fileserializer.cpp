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

#include "fileserializer.hpp"

#include <fstream>
#include <string>
#include <utility>

#include "idata.hpp"
#include "utilities/time.hpp"

FileSerializer::FileSerializer() : IProcessor() {
  add_option("path", path_, "Path (server-side) where to save data.");
  add_option("encoding", encoding_, "Binary or YAML encoding.");
  add_option("format", format_,
             "Data format (none, full, headeronly, streamheader, compact).");
  add_option("overwrite", overwrite_, "Overwrite existing files.");
  add_option(
      "throttle/enabled", throttle_,
      "Progressively drop incoming data packets if saving cannot keep up.");
  add_option("throttle/threshold", throttle_threshold_,
             "Buffer fill level (fraction between 0 and 1) at which throttling "
             "kicks in.");
  add_option("throttle/smooth", throttle_smooth_,
             "Smoothly changes throttle level as threshold is reached "
             "(value between 0 and 1).");
  add_option("preamble", preamble_, "Add YAML preamble to file.");
}

void FileSerializer::CreatePorts() {
  data_port_ =
      create_input_port<AnyType>("data", AnyType::Capabilities(),
                                 PortInPolicy(SlotRange(1, 256), false, 0));
}

void FileSerializer::Configure(const GlobalContext &context) {
  LOG(INFO) << "format: " << format_.to_yaml();
  LOG(INFO) << "encoding: " << encoding_.to_yaml();
  LOG(INFO) << "throttle enabled: " << throttle_();
  LOG(INFO) << "throttle threshold: " << throttle_threshold_();
  LOG(INFO) << "throttle smooth: " << throttle_smooth_();
}

void FileSerializer::Preprocess(ProcessingContext &context) {
  // create serializer object
  serializer_.reset(Serialization::serializer(encoding_(), format_()));

  // initialization
  packetid_.assign(data_port_->number_of_slots(), 0);
  upstream_buffer_size_.assign(data_port_->number_of_slots(), 0);
  nskipped_.assign(data_port_->number_of_slots(), 0);
  throttle_level_ = 0;

  // create output file streams
  streams_.clear();
  std::string path = context.resolve_path(path_(), "run");
  std::string address;
  std::string filename;
  std::unique_ptr<std::ostream> stream;

  for (int k = 0; k < data_port_->number_of_slots(); k++) {
    address = data_port_->slot(k)->upstream_address().string();
    filename = path + "/" + name() + "." + std::to_string(k) + "_" + address +
               "." + serializer_->extension();

    // if file exists
    if (!overwrite_() && path_exists(filename)) {
      throw ProcessingError("Output file " + filename + " exists.", name());
    }
    // try to open file
    stream = std::unique_ptr<std::ostream>(new std::ofstream(
        filename, std::ofstream::out | std::ofstream::binary));

    if (!stream->good()) {
      throw ProcessingError("Error opening output file " + filename + ".",
                            name());
    }

    if (preamble_()) {
      create_preamble(*stream.get(), k);
    }

    streams_.push_back(std::move(stream));

    LOG(DEBUG) << "Successfully opened output file for stream "
               << std::to_string(k) << " (" << address << ").";

    upstream_buffer_size_[k] = static_cast<unsigned int>(
        data_port_->slot(k)->upstream_policy().buffer_size());
  }
}

void FileSerializer::create_preamble(std::ostream &out, int slot) {
  YAML::Node node;
  node["creator"] = name();
  node["date"] = time_to_string(std::time(nullptr));
  node["version"] = static_cast<int>(Serialization::VERSION);
  node["interleaved"] = false;
  node["format"] = format_.to_yaml();
  node["encoding"] = encoding_.to_yaml();
  node["stream"] = slot;
  node["data"] =
      serializer_->DataDescription(data_port_->slot(slot)->GetDataPrototype());

  YAML::Emitter emit(out);
  emit << YAML::BeginDoc;
  emit << node;
  emit << YAML::EndDoc;
}

void FileSerializer::Process(ProcessingContext &context) {
  std::vector<AnyType::Data *> data;

  int nslots = data_port_->number_of_slots();
  uint64_t remainder;
  uint64_t nread;

  while (!context.terminated()) {
    for (int k = 0; k < nslots; ++k) {
      if (!data_port_->slot(k)->RetrieveDataAll(data)) {
        break;
      }

      nread = data_port_->slot(k)->status_read();

      if (nread == 0) {
        data_port_->slot(k)->ReleaseData();
        continue;
      }

      if (!throttle_()) {
        LOG_IF(WARNING, (nread > 0.5 * upstream_buffer_size_[k]))
            << name() << ": buffer is more than half full (stream " << k << ")";
        for (auto &it : data) {
          serializer_->Serialize(*(streams_[k]), it, k, packetid_[k]++,
                                 data_port_->slot(k)->upstream_address().processor(),
                                 data_port_->slot(k)->upstream_address().port(),
                                 data_port_->slot(k)->upstream_address().slot());
        }

      } else {
        // update throttle level
        throttle_level_ *= (1 - throttle_smooth_());
        if (nread > throttle_threshold_() * upstream_buffer_size_[k]) {
          throttle_level_ += throttle_smooth_();
        }

        remainder = std::floor(1.0 / (0.5 - std::abs(throttle_level_ - 0.5)));

        if (throttle_level_ == 0 ||
            (throttle_level_ < 0.5 && remainder > nread)) {
          // keep all
          for (auto &it : data) {
            serializer_->Serialize(*(streams_[k]), it, k, packetid_[k]++,
                                   data_port_->slot(k)->upstream_address().processor(),
                                   data_port_->slot(k)->upstream_address().port(),
                                   data_port_->slot(k)->upstream_address().slot());
          }
        } else if (throttle_level_ < 0.5) {
          // skip small fraction
          for (uint64_t n = 0; n < nread; ++n) {
            if (n % remainder == 0) {
              packetid_[k]++;
              nskipped_[k]++;
              continue;
            }
            serializer_->Serialize(*(streams_[k]), data[n], k, packetid_[k]++,
                                   data_port_->slot(k)->upstream_address().processor(),
                                   data_port_->slot(k)->upstream_address().port(),
                                   data_port_->slot(k)->upstream_address().slot());
          }
        } else if (throttle_level_ == 1 ||
                   (throttle_level_ >= 0.5 && remainder > nread)) {
          // skip all
          packetid_[k] += nread;
          nskipped_[k] += nread;
        } else {
          // eep small fraction
          for (uint64_t n = 0; n < nread; ++n) {
            if (n % remainder != 0) {
              packetid_[k]++;
              nskipped_[k]++;
              continue;
            }
            serializer_->Serialize(*(streams_[k]), data[n], k, packetid_[k]++,
                                   data_port_->slot(k)->upstream_address().processor(),
                                   data_port_->slot(k)->upstream_address().port(),
                                   data_port_->slot(k)->upstream_address().slot());
          }
        }
      }
      data_port_->slot(k)->ReleaseData();
    }
  }
}

void FileSerializer::Postprocess(ProcessingContext &context) {
  streams_.clear();  // forces destruction and closing of resources
  serializer_.reset();

  for (SlotType k = 0; k < data_port_->number_of_slots(); k++) {
    if (nskipped_[k] != 0) {
      LOG(UPDATE) << name() << ": stream " << k << ": received " << packetid_[k]
                  << " data packets, of which " << nskipped_[k]
                  << " were not saved.";
    } else {
      LOG(UPDATE) << name() << ": stream " << k << ": received and saved "
                  << packetid_[k] << " data packets.";
    }
  }
}

REGISTERPROCESSOR(FileSerializer)
