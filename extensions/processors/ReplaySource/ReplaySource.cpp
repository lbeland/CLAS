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

#include "ReplaySource.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "logging/log.hpp"

namespace fs = std::filesystem;

// DtypeSize — byte size of one element for a given dtype string
std::size_t ReplaySource::DtypeSize(const std::string &dtype)
{
    if (dtype == "int8" || dtype == "uint8")
        return 1;
    if (dtype == "int16" || dtype == "uint16")
        return 2;
    if (dtype == "int32" || dtype == "uint32" || dtype == "float32")
        return 4;
    if (dtype == "int64" || dtype == "uint64" || dtype == "float64")
        return 8;
    throw std::runtime_error("ReplaySource: unknown dtype '" + dtype + "'");
}

// ParseDataEntry — parse one YAML "data" list entry, e.g. "signal float32 (32,64)"
FieldDescriptor ReplaySource::ParseDataEntry(const std::string &entry)
{
    static const std::regex kPattern(R"(^(.+?)\s+(\w+)\s*\(([^)]*)\)\s*$)");
    std::smatch m;
    if (!std::regex_match(entry, m, kPattern))
    {
        throw std::runtime_error(
            "ReplaySource: cannot parse data entry '" + entry + "'");
    }

    FieldDescriptor fd;
    fd.name = m[1].str();
    fd.dtype = m[2].str();
    for (auto &c : fd.dtype)
        c = static_cast<char>(std::tolower(c));

    std::istringstream dim_ss(m[3].str());
    std::string tok;
    while (std::getline(dim_ss, tok, ','))
    {
        if (!tok.empty())
            fd.dims.push_back(std::stoi(tok));
    }
    if (fd.dims.empty())
    {
        throw std::runtime_error(
            "ReplaySource: missing dimensions in entry '" + entry + "'");
    }

    fd.n_items = 1;
    for (int d : fd.dims)
        fd.n_items *= static_cast<std::size_t>(d);
    fd.byte_size = fd.n_items * DtypeSize(fd.dtype);
    fd.offset = 0; // filled in by LoadFile
    return fd;
}

// ResolveFilePath
// Priority:
//   1. file_ option set explicitly → use verbatim.
//   2. Otherwise search <resolved_path>/ for a file matching
//      "*.<slot_>_*.bin"  (the naming convention from FileSerializer).
std::string ReplaySource::ResolveFilePath(const std::string path) const
{
    if (!file_().empty())
    {
        return file_();
    }

    fs::path run_dir(path);
    if (!fs::is_directory(run_dir))
    {
        throw std::runtime_error(
            "ReplaySource: could not resolve replay folder from path='" +
            path_() + "'. Not a directory.");
    }

    auto ends_with = [](const std::string &value, const std::string &suffix)
    {
        return value.size() >= suffix.size() &&
               value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    const std::string slot_str = std::to_string(slot_());
    const std::string eeg_suffix = ".eeg." + slot_str + ".bin";

    std::vector<std::string> eeg_matches;
    std::vector<std::string> slot_matches;

    for (const auto &entry : fs::recursive_directory_iterator(run_dir))
    {
        if (!entry.is_regular_file())
            continue;

        const std::string fname = entry.path().filename().string();

        // Preferred pattern: SerializerX.<slot>_UDPSource.eeg.<slot>.bin
        if (ends_with(fname, eeg_suffix) &&
            fname.find("_UDPSource") != std::string::npos)
        {
            eeg_matches.push_back(entry.path().string());
            continue;
        }

        // Backward-compatible fallback: <processor_name>.<slot>_<upstream>.bin
        std::regex slot_pat(".*\\." + slot_str + "_.*\\.bin$");
        if (std::regex_match(fname, slot_pat))
        {
            slot_matches.push_back(entry.path().string());
        }
    }

    if (!eeg_matches.empty())
    {
        std::sort(eeg_matches.begin(), eeg_matches.end());
        return eeg_matches.front();
    }

    if (!slot_matches.empty())
    {
        std::sort(slot_matches.begin(), slot_matches.end());
        return slot_matches.front();
    }

    throw std::runtime_error(
        "ReplaySource: no serialized file found for slot " + slot_str +
        " in '" + run_dir.string() + "'");
}

// LoadFile — read YAML header + binary payload into memory
void ReplaySource::LoadFile(const std::string &filepath)
{
    std::ifstream f(filepath, std::ios::binary);
    if (!f)
    {
        throw std::runtime_error(
            "ReplaySource: cannot open file '" + filepath + "'");
    }

    std::vector<std::uint8_t> blob(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>());

    // 1. Locate YAML header terminator "...\n" or "...\r\n"
    std::size_t header_end = std::string::npos;
    for (const auto &marker : {std::string("...\n"), std::string("...\r\n")})
    {
        auto it = std::search(blob.begin(), blob.end(),
                              marker.begin(), marker.end());
        if (it != blob.end())
        {
            header_end = static_cast<std::size_t>(
                             std::distance(blob.begin(), it)) +
                         marker.size();
            break;
        }
    }
    if (header_end == std::string::npos)
    {
        throw std::runtime_error(
            "ReplaySource: YAML header terminator not found in '" +
            filepath + "'");
    }

    // 2. Parse YAML header
    const std::string header_str(blob.begin(),
                                 blob.begin() + static_cast<std::ptrdiff_t>(header_end));
    YAML::Node header = YAML::Load(header_str);

    if (!header["data"] || !header["data"].IsSequence())
    {
        throw std::runtime_error(
            "ReplaySource: 'data' key missing or not a sequence in header");
    }

    // 3. Build field layout
    layout_.clear();
    record_size_ = 0;

    for (const auto &node : header["data"])
    {
        FieldDescriptor fd = ParseDataEntry(node.as<std::string>());
        fd.offset = record_size_;
        record_size_ += fd.byte_size;
        layout_.push_back(fd);
    }

    // 4. Locate the "signal" field (written by MultiChannelData)
    signal_offset_ = 0;
    bool found_signal = false;
    for (const auto &fd : layout_)
    {
        if (fd.name == "signal" || fd.name == "scalar_data")
        {
            signal_offset_ = fd.offset;
            signal_dtype_ = fd.dtype;
            if (fd.dtype != "float32" && fd.dtype != "float64")
            {
                throw std::runtime_error(
                    "ReplaySource: unsupported signal dtype '" + fd.dtype +
                    "' in file '" + filepath + "' (expected float32 or float64)");
            }

            if (fd.dims.size() >= 2)
            {
                const auto file_ch = static_cast<unsigned int>(fd.dims[0]);
                const auto file_ns = static_cast<unsigned int>(fd.dims[1]);
                if (file_ch != nchannels_() || file_ns != nsamples_())
                {
                    LOG(WARNING) << name()
                                 << ": file signal shape (" << file_ch << "," << file_ns
                                 << ") does not match configured nchannels=" << nchannels_()
                                 << " nsamples=" << nsamples_()
                                 << ". Using file dimensions.";
                }
            }
            found_signal = true;
            break;
        }
    }
    if (!found_signal)
    {
        throw std::runtime_error(
            "ReplaySource: no 'signal' or 'scalar_data' field in file '" +
            filepath + "'");
    }

    // 5. Store payload
    payload_.assign(blob.begin() + static_cast<std::ptrdiff_t>(header_end),
                    blob.end());

    if (record_size_ == 0)
    {
        throw std::runtime_error("ReplaySource: record size is zero");
    }

    n_records_ = payload_.size() / record_size_;
    payload_.resize(n_records_ * record_size_); // trim to whole records

    LOG(INFO) << name() << ": loaded " << n_records_ << " records ("
              << record_size_ << " bytes/record) from '" << filepath << "'";
}

ReplaySource::ReplaySource() : IProcessor(PRIORITY_HIGH)
{
    add_option("path", path_, "Replay folder or run name. Run names resolve via _last_run_group/<run> first, then results/<run>; empty defaults to results.");
    add_option("file", file_, "Optional explicit path to a recorded EEG stream file.");
    add_option("slot", slot_, "Recorded slot to replay when a processor wrote multiple streams.");
    add_option("loop", loop_, "Restart replay from the beginning after the last packet.");
    add_option("real_time", real_time_, "Replay using recorded timing instead of emitting as fast as possible.");
    add_option("speed_factor", speed_factor_, "Playback speed multiplier.");
    add_option("n_messages", n_messages_, "Number of packets to replay (-1 = all available packets).");
    add_option("nchannels", nchannels_, "Number of EEG channels to replay.");
    add_option("nsamples", nsamples_, "Number of samples per packet.");
    add_option("fs", fs_, "Replay sample frequency (Hz).");
}

void ReplaySource::CreatePorts()
{
    data_out_port_ = create_output_port<MultiChannelType<double>>(
        "out",
        MultiChannelType<double>::Parameters(nchannels_(), nsamples_(), fs_()),
        PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));
}

void ReplaySource::CompleteStreamInfo()
{
    data_out_port_->slot(0)->streaminfo().set_parameters(
        MultiChannelType<double>::Parameters(nchannels_(), nsamples_(), fs_()));
    data_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());
}

void ReplaySource::Prepare(GlobalContext &context)
{
    // File loading is deferred to Preprocess to allow dynamic path resolution.
}

void ReplaySource::Preprocess(ProcessingContext &context)
{
    current_record_ = 0;
    emitted_ = 0;

    // Pre-compute the expected wall-clock gap between consecutive packets.
    // Gap [ns] = nsamples / fs * 1e9 / speed_factor
    inter_packet_ns_ =
        (static_cast<double>(nsamples_()) / fs_()) * 1e9 / speed_factor_();

    std::string path_resolved = context.resolve_path(path_(), "lastrungroup");
    const std::string filepath = ResolveFilePath(path_resolved);
    LoadFile(filepath);

    LOG(INFO) << name() << " configured for replay."
              << " path=" << filepath
              << " file=" << file_()
              << " slot=" << slot_()
              << " nchannels=" << nchannels_()
              << " nsamples=" << nsamples_()
              << " fs=" << fs_()
              << " real_time=" << real_time_()
              << " speed_factor=" << speed_factor_()
              << " inter_packet_gap=" << inter_packet_ns_ / 1e6 << " ms"
              << " loop=" << loop_()
              << " n_records=" << n_records_;
}

void ReplaySource::Process(ProcessingContext &context)
{
    if (n_records_ == 0)
    {
        LOG(WARNING) << name() << ": no records to replay.";
        return;
    }

    MultiChannelType<double>::Data *data_out = nullptr;
    auto *out_slot = data_out_port_->slot(0);
    const bool signal_is_float32 = (signal_dtype_ == "float32");

    const std::int64_t max_packets = n_messages_();

    // Record the wall-clock time just before we start emitting so the first
    // packet is sent immediately and subsequent packets are paced from there.
    last_emit_time_ = Clock::now();

    while (!context.terminated())
    {
        if (max_packets >= 0 && emitted_ >= max_packets)
        {
            break;
        }

        if (current_record_ >= n_records_)
        {
            if (loop_())
            {
                current_record_ = 0;
                LOG(DEBUG) << name() << ": looping replay from the beginning.";
            }
            else
            {
                LOG(INFO) << name() << ": reached end of recorded data.";
                break;
            }
        }

        // Real-time pacing: sleep until the next packet is due
        if (real_time_())
        {
            const TimePoint target =
                last_emit_time_ +
                std::chrono::nanoseconds(
                    static_cast<std::int64_t>(inter_packet_ns_));

            const TimePoint now = Clock::now();
            if (target > now)
            {
                std::this_thread::sleep_until(target);
            }
        }

        data_out = out_slot->ClaimData(false);

        // Pointer to the start of the current record in the payload
        const std::uint8_t *record_ptr = payload_.data() + current_record_ * record_size_;

        // Copy signal samples: layout is [nchannels][nsamples], stored as either
        // float32 (recordings made before the float->double conversion) or
        // float64 (current format); either way it is mapped to double on read.
        const unsigned int nch = nchannels_();
        const unsigned int ns = nsamples_();
        if (signal_is_float32)
        {
            const float *src = reinterpret_cast<const float *>(record_ptr + signal_offset_);
            for (unsigned int ch = 0; ch < nch; ++ch)
            {
                for (unsigned int s = 0; s < ns; ++s)
                {
                    data_out->set_data_sample(s, ch, static_cast<double>(src[ch * ns + s]));
                }
            }
        }
        else
        {
            const double *src = reinterpret_cast<const double *>(record_ptr + signal_offset_);
            for (unsigned int ch = 0; ch < nch; ++ch)
            {
                for (unsigned int s = 0; s < ns; ++s)
                {
                    data_out->set_data_sample(s, ch, src[ch * ns + s]);
                }
            }
        }

        // Propagate hardware timestamp when available
        const FieldDescriptor *hw_ts_fd = nullptr;
        for (const auto &fd : layout_)
        {
            if (fd.name == "hardware_ts")
            {
                hw_ts_fd = &fd;
                break;
            }
        }
        if (hw_ts_fd != nullptr)
        {
            std::uint64_t hw_ts = 0;
            std::memcpy(&hw_ts, record_ptr + hw_ts_fd->offset, sizeof(hw_ts));
            data_out->set_hardware_timestamp(hw_ts);
        }

        last_emit_time_ = Clock::now();
        data_out->set_source_timestamp(last_emit_time_);

        out_slot->PublishData();

        ++current_record_;
        ++emitted_;
    }
    LOG(INFO) << name() << " stopped working";
}

void ReplaySource::Postprocess(ProcessingContext &context)
{
    LOG(INFO) << name() << ": Total replay packets emitted: " << emitted_;
}

REGISTERPROCESSOR(ReplaySource)
