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

#include "ReplaySourceClient.hpp"

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

// ---------------------------------------------------------------------------
// dtype helpers
// ---------------------------------------------------------------------------

std::size_t ReplaySourceClient::DtypeSize(const std::string &dtype)
{
    if (dtype == "int8" || dtype == "uint8")
        return 1;
    if (dtype == "int16" || dtype == "uint16")
        return 2;
    if (dtype == "int32" || dtype == "uint32" || dtype == "float32")
        return 4;
    if (dtype == "int64" || dtype == "uint64" || dtype == "float64")
        return 8;
    throw std::runtime_error("ReplaySourceClient: unknown dtype '" + dtype + "'");
}

// ---------------------------------------------------------------------------
// Parse one YAML "data" list entry, e.g. "signal float32 (32,64)"
// ---------------------------------------------------------------------------

FieldDescriptor ReplaySourceClient::ParseDataEntry(const std::string &entry)
{
    // Pattern: <name> <dtype> (<d0>[,<d1>,...])
    static const std::regex kPattern(R"(^(.+?)\s+(\w+)\s*\(([^)]*)\)\s*$)");
    std::smatch m;
    if (!std::regex_match(entry, m, kPattern))
    {
        throw std::runtime_error(
            "ReplaySourceClient: cannot parse data entry '" + entry + "'");
    }

    FieldDescriptor fd;
    fd.name = m[1].str();
    // Normalise to lowercase, e.g. "Float32" -> "float32"
    fd.dtype = m[2].str();
    for (auto &c : fd.dtype)
        c = static_cast<char>(std::tolower(c));

    // Parse comma-separated dimensions
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
            "ReplaySourceClient: missing dimensions in entry '" + entry + "'");
    }

    fd.n_items = 1;
    for (int d : fd.dims)
        fd.n_items *= static_cast<std::size_t>(d);
    fd.byte_size = fd.n_items * DtypeSize(fd.dtype);
    fd.offset = 0; // filled in by LoadFile
    return fd;
}

// ---------------------------------------------------------------------------
// ResolveFilePath
//
// Priority:
//   1. file_ option set explicitly → use verbatim.
//   2. Otherwise search <resolved_path>/ for a file matching
//      "*.<slot_>_*.bin"  (the naming convention from FileSerializer).
// ---------------------------------------------------------------------------

std::string ReplaySourceClient::ResolveFilePath(const std::string path) const
{
    if (!file_().empty())
    {
        return file_();
    }

    // path_ may carry a "run://" prefix; strip it to keep local resolution
    // explicit and deterministic.
    std::string input = path;

    std::vector<fs::path> candidate_dirs;
    const fs::path as_path(input);

    // If an existing directory is given explicitly, use it directly.
    if (fs::is_directory(as_path))
    {
        candidate_dirs.push_back(as_path);
    }

    fs::path run_dir;
    for (const auto &candidate : candidate_dirs)
    {
        if (fs::is_directory(candidate))
        {
            run_dir = candidate;
            break;
        }
    }

    if (run_dir.empty())
    {
        throw std::runtime_error(
            "ReplaySourceClient: could not resolve replay folder from path='" +
            path_() + "'. Tried _last_run_group/<run> and results/<run>.");
    }

    auto ends_with = [](const std::string &value, const std::string &suffix)
    {
        return value.size() >= suffix.size() &&
               value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    const std::string slot_str = std::to_string(slot_());
    const std::string sourceclient_suffix = ".out." + slot_str + ".bin";

    std::vector<std::string> sourceclient_matches;
    std::vector<std::string> slot_matches;

    for (const auto &entry : fs::recursive_directory_iterator(run_dir))
    {
        if (!entry.is_regular_file())
            continue;

        const std::string fname = entry.path().filename().string();

        // Preferred serializer pattern:
        //   SerializerX.<slot>_SourceClient.out.<slot>.bin
        if (ends_with(fname, sourceclient_suffix) &&
            fname.find("_SourceClient") != std::string::npos)
        {
            sourceclient_matches.push_back(entry.path().string());
            continue;
        }

        // Backward-compatible fallback pattern:
        //   <processor_name>.<slot>_<upstream>.bin
        std::regex slot_pat(".*\\." + slot_str + "_.*\\.bin$");
        if (std::regex_match(fname, slot_pat))
        {
            slot_matches.push_back(entry.path().string());
        }
    }

    if (!sourceclient_matches.empty())
    {
        std::sort(sourceclient_matches.begin(), sourceclient_matches.end());
        return sourceclient_matches.front();
    }

    if (!slot_matches.empty())
    {
        std::sort(slot_matches.begin(), slot_matches.end());
        return slot_matches.front();
    }

    throw std::runtime_error(
        "ReplaySourceClient: no serialized file found for slot " + slot_str +
        " in '" + run_dir.string() + "'");

}

// ---------------------------------------------------------------------------
// LoadFile — read header + binary payload into memory
// ---------------------------------------------------------------------------

void ReplaySourceClient::LoadFile(const std::string &filepath)
{
    std::ifstream f(filepath, std::ios::binary);
    if (!f)
    {
        throw std::runtime_error(
            "ReplaySourceClient: cannot open file '" + filepath + "'");
    }

    // Slurp entire file
    std::vector<std::uint8_t> blob(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>());

    // ------------------------------------------------------------------
    // 1. Locate YAML header terminator "...\n" or "...\r\n"
    // ------------------------------------------------------------------
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
            "ReplaySourceClient: YAML header terminator not found in '" +
            filepath + "'");
    }

    // ------------------------------------------------------------------
    // 2. Parse YAML header
    // ------------------------------------------------------------------
    const std::string header_str(blob.begin(),
                                 blob.begin() + static_cast<std::ptrdiff_t>(header_end));
    YAML::Node header = YAML::Load(header_str);

    if (!header["data"] || !header["data"].IsSequence())
    {
        throw std::runtime_error(
            "ReplaySourceClient: 'data' key missing or not a sequence in header");
    }

    // ------------------------------------------------------------------
    // 3. Build field layout
    // ------------------------------------------------------------------
    layout_.clear();
    record_size_ = 0;

    for (const auto &node : header["data"])
    {
        FieldDescriptor fd = ParseDataEntry(node.as<std::string>());
        fd.offset = record_size_;
        record_size_ += fd.byte_size;
        layout_.push_back(fd);
    }

    // ------------------------------------------------------------------
    // 4. Locate the "signal" field (written by MultiChannelData)
    // ------------------------------------------------------------------
    signal_offset_ = 0;
    bool found_signal = false;
    for (const auto &fd : layout_)
    {
        if (fd.name == "signal" || fd.name == "scalar_data")
        {
            signal_offset_ = fd.offset;

            // Validate against configured nchannels / nsamples when available
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
            "ReplaySourceClient: no 'signal' or 'scalar_data' field in file '" +
            filepath + "'");
    }

    // ------------------------------------------------------------------
    // 5. Store payload
    // ------------------------------------------------------------------
    payload_.assign(blob.begin() + static_cast<std::ptrdiff_t>(header_end),
                    blob.end());

    if (record_size_ == 0)
    {
        throw std::runtime_error("ReplaySourceClient: record size is zero");
    }

    n_records_ = payload_.size() / record_size_;
    // Trim to whole records
    payload_.resize(n_records_ * record_size_);

    LOG(INFO) << name() << ": loaded " << n_records_ << " records ("
              << record_size_ << " bytes/record) from '" << filepath << "'";
}

// ===========================================================================
// IProcessor interface
// ===========================================================================

ReplaySourceClient::ReplaySourceClient() : IProcessor(PRIORITY_HIGH)
{
    add_option("path", path_, "Replay folder or run name. Run names resolve via _last_run_group/<run> first, then results/<run>; empty defaults to results.");
    add_option("file", file_, "Optional file path to a specific recorded EEG stream.");
    add_option("slot", slot_, "Recorded slot to replay when a processor wrote multiple streams.");
    add_option("loop", loop_, "Restart replay from the beginning after the last packet.");
    add_option("real_time", real_time_, "Replay using recorded timing instead of emitting as fast as possible.");
    add_option("speed_factor", speed_factor_, "Playback speed multiplier.");
    add_option("n_messages", n_messages_, "Number of packets to replay (-1 = all available packets).");
    add_option("nchannels", nchannels_, "Number of EEG channels to replay.");
    add_option("nsamples", nsamples_, "Number of samples per packet.");
    add_option("fs", fs_, "Replay sample frequency.");
}

void ReplaySourceClient::CreatePorts()
{
    data_out_port_ = create_output_port<MultiChannelType<float>>(
        "out",
        MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()),
        PortOutPolicy(SlotRange(1), 200, WaitStrategy::kBlockingStrategy));
}

void ReplaySourceClient::CompleteStreamInfo()
{
    data_out_port_->slot(0)->streaminfo().set_parameters(
        MultiChannelType<float>::Parameters(nchannels_(), nsamples_(), fs_()));
    data_out_port_->slot(0)->streaminfo().set_stream_rate(fs_());
}

void ReplaySourceClient::Prepare(GlobalContext &context)
{
    // Nothing to prepare; file loading is deferred to Preprocess to allow
    // dynamic path resolution based on the run context.
}

void ReplaySourceClient::Preprocess(ProcessingContext &context)
{
    send_times.clear();
    current_record_ = 0;

    // Pre-compute the expected wall-clock gap between consecutive packets.
    // Each packet carries nsamples_ samples at fs_ Hz.
    // Gap [ns] = nsamples / fs * 1e9 / speed_factor
    inter_packet_ns_ =
        (static_cast<double>(nsamples_()) / fs_()) * 1e9 / speed_factor_();

    std::string path_resolved = context.resolve_path(path_(),"lastrungroup");

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
              << " loop=" << loop_()
              << " n_records=" << n_records_;
}

// ---------------------------------------------------------------------------
// Process — main replay loop
// ---------------------------------------------------------------------------

void ReplaySourceClient::Process(ProcessingContext &context)
{
    if (n_records_ == 0)
    {
        LOG(WARNING) << name() << ": no records to replay.";
        return;
    }

    MultiChannelType<float>::Data *data_out = nullptr;

    // Maximum number of packets to emit (−1 = unlimited)
    const std::int64_t max_packets = n_messages_();
    std::int64_t emitted = 0;

    // Grab the output slot once
    auto *out_slot = data_out_port_->slot(0);

    // Record the wall-clock time just before we start emitting so the first
    // packet is sent immediately and subsequent packets are paced from there.
    last_emit_time_ = Clock::now();

    while (!context.terminated())
    {
        // ----------------------------------------------------------------
        // Check message limit
        // ----------------------------------------------------------------
        if (max_packets >= 0 && emitted >= max_packets)
        {
            break;
        }

        // ----------------------------------------------------------------
        // Loop / end-of-file handling
        // ----------------------------------------------------------------
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

        // ----------------------------------------------------------------
        // Real-time pacing: sleep until the next packet is due
        // ----------------------------------------------------------------
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

        // ----------------------------------------------------------------
        // Retrieve and fill the output data object
        // ----------------------------------------------------------------
        data_out = out_slot->ClaimData(false);

        // Pointer to the start of the current record in the payload
        const std::uint8_t *record_ptr = payload_.data() + current_record_ * record_size_;

        // Copy signal samples: layout is [nchannels][nsamples] of float32,
        // stored contiguously in the record at signal_offset_.
        const float *src = reinterpret_cast<const float *>(
            record_ptr + signal_offset_);

        const unsigned int nch = nchannels_();
        const unsigned int ns = nsamples_();

        for (unsigned int ch = 0; ch < nch; ++ch)
        {
            for (unsigned int s = 0; s < ns; ++s)
            {
                data_out->set_data_sample(s, ch, src[ch * ns + s]);
            }
        }

        // Propagate hardware and source timestamp when available
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

        TimePoint src_time_point;
        src_time_point = Clock::now();
        data_out->set_source_timestamp(src_time_point);

        // ----------------------------------------------------------------
        // Publish the data object
        // ----------------------------------------------------------------
        send_times.push_back(src_time_point);

        out_slot->PublishData();

        ++current_record_;
        ++emitted;
    }
}

void ReplaySourceClient::Postprocess(ProcessingContext &context)
{
    std::ostringstream statistic_print;
    statistic_print << "\n ---------------- \n Total replay packets emitted: "
                    << send_times.size();
    std::cout << statistic_print.str();
}

REGISTERPROCESSOR(ReplaySourceClient)