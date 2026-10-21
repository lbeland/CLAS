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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

#include "../utilities/zmqutil.hpp"
#include "customsink.hpp"

namespace {
// run start/end as nanoseconds since epoch of the g3log clock (0 = unset)
std::atomic<int64_t> run_start_ns{0};
std::atomic<int64_t> run_end_ns{0};

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

// Sinks format messages asynchronously, so the elapsed time is computed from the
// message's own timestamp rather than from the current time.
std::string ElapsedSinceRunStart(const g3::LogMessage& msg) {
    const int64_t start = run_start_ns.load();
    const int64_t end = run_end_ns.load();
    const int64_t ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           msg._timestamp.time_since_epoch())
                           .count();
    if (start == 0 or ts < start or (end != 0 and ts > end)) {
        return "-";
    }
    return std::to_string((ts - start) / 1000000000) + "s";
}

// e.g. "2026/10/21 11:27:35.812 50s\t"
std::string LogHeader(const g3::LogMessage& msg) {
    return msg.timestamp("%Y/%m/%d %H:%M:%S.%f3") + " " + ElapsedSinceRunStart(msg) + "\t";
}

std::string FileLogDetails(const g3::LogMessage& msg) {
    return LogHeader(msg) + msg.level() + " [" + msg.file() + "->" + msg.function() + ":" +
           msg.line() + "]\t";
}
}  // namespace

void SetRunStart() {
    run_end_ns.store(0);
    run_start_ns.store(now_ns());
}

void SetRunEnd() { run_end_ns.store(now_ns()); }

FileSinkFixed::FileSinkFixed(const std::string& path) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    _out.open(path, std::ios::trunc);
    if (!_out.is_open()) {
        std::cerr << "FileSinkFixed: cannot open log file: " << path << std::endl;
    }
}

void FileSinkFixed::ReceiveLogMessage(g3::LogMessageMover message) {
    _out << message.get().toString(FileLogDetails) << std::flush;
}

std::string ScreenSink::FormatMessage(const LEVELS& level, g3::LogMessage& msg) {
    if (level.value == DEBUG.value or level.value >= WARNING.value) {
        std::string out;
        out.append("\n" + LogHeader(msg) + msg.level() + " [" + msg.file() +
                   " L: " + msg.line() + "]\t");
        out.append(msg.message());
        return out;
    } else if (msg.wasFatal()) {
        return msg.message();
    } else {
        std::string out;
        out.append("\n" + LogHeader(msg) + msg.level() + "\t" + msg.message());
        return out;
    }
}

ScreenSink::FG_Color ScreenSink::GetColor(const LEVELS& level) const {
    if (level.value == WARNING.value) {
        return YELLOW;
    }
    if (level.value == DEBUG.value) {
        return GREEN;
    }
    if (level.value >= WARNING.value) {
        return RED;
    }

    return WHITE;
}

void ScreenSink::ReceiveLogMessage(g3::LogMessageMover message) {
    ScreenSink::FG_Color color;
    color = GetColor(message.get()._level);
    std::cout << "\033[" << color << "m" << FormatMessage(message.get()._level, message.get())
              << "\033[m" << std::flush;
}

ZMQSink::ZMQSink(zmq::context_t& context, int port) {
    // set up PUB
    publisher = new zmq::socket_t(context, ZMQ_PUB);

    char buffer[15];
    snprintf(buffer, 14, "tcp://*:%d", port);
    publisher->bind(buffer);
}

ZMQSink::~ZMQSink() {
    // clean up PUB
    delete publisher;
}

std::deque<std::string> ZMQSink::FormatMessage(g3::LogMessage& msg) {
    std::deque<std::string> out;
    out.push_back(msg.level());
    out.push_back(msg.timestamp());

    if (!msg.wasFatal()) {
        out.push_back(msg.message());
        if (msg.level() == "DEBUG") {
            out.push_back("[" + msg.file() + " L: " + msg.line() + "]");
        }
    }
    return out;
}

void ZMQSink::ReceiveLogMessage(g3::LogMessageMover message) {
    zmq_frames msg = FormatMessage(message.get());
    s_send_multi(*publisher, msg);
}
