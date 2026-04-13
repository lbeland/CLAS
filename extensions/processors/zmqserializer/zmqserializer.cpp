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

#include "zmqserializer.hpp"

#include <string>
#include <utility>

#include "idata.hpp"
#include "utilities/zmqutil.hpp"

ZMQSerializer::ZMQSerializer() : IProcessor() {
  add_option(
      "port", port_,
      "Local network port for data streaming. If interleave is false, "
      "then this is the first port in a sequence of ports used for streaming.");

  add_option("encoding", encoding_, "Binary or YAML encoding.");
  add_option("format", format_,
             "Data format (none, full, headeronly, streamheader, compact).");

  add_option("interleave", interleave_,
             "Interleave data streams from all input slots and stream to "
             "single network port.");
  add_option("n_messages", n_messages_, "Number of packets to receive (-1 = infinite).");
}

void ZMQSerializer::CreatePorts() {
  data_port_ =
      create_input_port<AnyType>("in", AnyType::Capabilities(),
                                 PortInPolicy(SlotRange(1, 256), false));
}

void ZMQSerializer::Preprocess(ProcessingContext &context) {
  std::string address;
  sockets_.clear();

  if (interleave_()) {
    sockets_.push_back(std::make_unique<zmq::socket_t>(context.run().global().zmq(), ZMQ_PUB));
    address = "tcp://*:" + std::to_string(port_());
    sockets_.back()->bind(address.c_str());
  } else {
    for (int k = 0; k < data_port_->number_of_slots(); ++k) {
      sockets_.push_back(std::make_unique<zmq::socket_t>(context.run().global().zmq(), ZMQ_PUB));
      address = "tcp://*:" + std::to_string(port_() + k);
      sockets_.back()->bind(address.c_str());
    }
  }

  serializer_.reset(Serialization::serializer(encoding_(), format_()));
  packetid_.assign(data_port_->number_of_slots(), 0);
}

void ZMQSerializer::Process(ProcessingContext &context) {
  std::vector<typename AnyType::Data *> data;
  unsigned int idx = 0;
  std::stringstream buffer;

  while (!context.terminated()) {
    for (int k = 0; k < data_port_->number_of_slots(); ++k) {
      if (!data_port_->slot(k)->RetrieveDataAll(data)) {
        break;
      }

      if (!interleave_()) {
        idx = k;
      }

      for (auto &it : data) {
        buffer.str("");
        buffer.clear();

        if (serializer_->Serialize(buffer, it, k, packetid_[k]++,
                                   data_port_->slot(k)->upstream_address().processor(),
                                   data_port_->slot(k)->upstream_address().port(),
                                   data_port_->slot(k)->upstream_address().slot())) {
          if (!s_send(*(sockets_[idx]), buffer.str())) {
            LOG(DEBUG) << "failed to send zmq message.";
          }
        } else {
          LOG(WARNING) << name() << ": Unable to serialize data stream " << k;
        }
      }
      data_port_->slot(k)->ReleaseData();
    }
    
    // Check if all slots have received the specified number of messages (if n_messages_ is set)
    if (n_messages_() != -1) {
      bool all_slots_done = true;
      for (int k = 0; k < data_port_->number_of_slots(); ++k) {
        if (packetid_[k] < static_cast<size_t>(n_messages_())) {
          all_slots_done = false;
          break;
        }
      }
      if (all_slots_done) {
        break;
      }
    }

  }
}

void ZMQSerializer::Postprocess(ProcessingContext &context) {
  sockets_.clear();
  serializer_.reset();

  for (SlotType k = 0; k < data_port_->number_of_slots(); k++) {
    LOG(UPDATE) << name() << ": stream " << k
                << ": received and serialized over network " << packetid_[k]
                << " data packets.";
  }
}

REGISTERPROCESSOR(ZMQSerializer)
