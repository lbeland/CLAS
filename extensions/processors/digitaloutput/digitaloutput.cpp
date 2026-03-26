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

#include "digitaloutput.hpp"
#include "dio/dummydio.hpp"

#include <iostream>

DigitalOutput::DigitalOutput() {
  add_option("pulse width", pulse_width_, "duration of digital output pulse in microseconds");
  add_option("device/type", device_type_, "Only dummy type implemented yet");
  add_option("device/nchannels", nchannels_, "Number of digital channel on the device.");
  add_option("protocols", protocols_yaml_, "");
  add_option("event logging", event_log_, "Log message (UPDATE level) if true");
}
void DigitalOutput::Configure(const GlobalContext &context) {

  if (device_type_() == "dummy") {
    device_ = std::unique_ptr<DigitalDevice>(new DummyDIO(nchannels_()));
  } else {
    throw ProcessingConfigureError("No valid digital output device specified.",
                                   name());
  }

  LOG(INFO) << "Opened digital output device " << device_->description() << ".";

  for (auto const &it : protocols_yaml_()) {
    protocols_[it.first] = std::unique_ptr<DigitalOutputProtocol>(
        new DigitalOutputProtocol(device_->nchannels(), pulse_width_()));
    for (auto const &it2 : it.second) {
      if (it2.first == "toggle") {
        protocols_[it.first]->set_mode(it2.second, DigitalOutputMode::TOGGLE);
      } else if (it2.first == "high") {
        protocols_[it.first]->set_mode(it2.second, DigitalOutputMode::HIGH);
      } else if (it2.first == "low") {
        protocols_[it.first]->set_mode(it2.second, DigitalOutputMode::LOW);
      } else if (it2.first == "pulse") {
        protocols_[it.first]->set_mode(it2.second, DigitalOutputMode::PULSE);
      }
    }
  }

  LOG(INFO) << name() << ". There are " << protocols_.size()
            << " configured output protocols.";
}

void DigitalOutput::CreatePorts() {
  data_in_port_ = create_input_port<EventType>(EventType::Capabilities(),
                                               PortInPolicy(SlotRange(1)));
}

void DigitalOutput::Process(ProcessingContext &context) {

  EventType::Data *data_in = nullptr;

  while (!context.terminated()) {

    if (!data_in_port_->slot(0)->RetrieveData(data_in)) {
      break;
    }

    // select and execute protocol based on event name
    if (protocols_.count(data_in->event()) > 0) {

        try {
          protocols_[data_in->event()]->execute(*device_);

          LOG(UPDATE) << name() << ". Protocol executed for "
                      << data_in->event() << " event.";
        } catch (DigitalDeviceError &e) {
          LOG(WARNING) << ". Could not execute protocol for event ("
                       << data_in->event() << ") : " << e.what();
        }

    }

    data_in_port_->slot(0)->ReleaseData();
  }
}

REGISTERPROCESSOR(DigitalOutput)
