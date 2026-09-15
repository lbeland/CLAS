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

#include "iprocessor.hpp"
#include "multichanneldata/multichanneldata.hpp"
#include "options/options.hpp"
#include "utilities/time.hpp"

class SimulatedSource : public IProcessor
{
public:
    SimulatedSource();

    void CreatePorts() override;
    void CompleteStreamInfo() override;
    void Preprocess(ProcessingContext &context) override;
    void Process(ProcessingContext &context) override;
    void Postprocess(ProcessingContext &context) override;

protected:
    // Data ports
    PortOut<MultiChannelType<double>> *data_out_port_;
    PortOut<MultiChannelType<double>> *meta_out_port_;

    // Options
    options::String path_{"run://"};
    options::Double fs_{10000.0};
    options::Double carrier_amplitude_{3.0};
    options::Double carrier_frequency_{8.0};
    // Signal mode, mutually exclusive:
    //   none | amplitude | phase | phase_jump | freq_step | chirp
    options::String modulation_type_{"phase"};
    options::Double modulation_amplitude_{1.0};  // amplitude/phase: modulation depth
    options::Double modulation_frequency_{0.1};  // amplitude/phase: modulation rate (Hz)
    options::Double mod_period_s_{5.0};          // phase_jump/freq_step: unperturbed segment length (s)
    options::Double phase_jump_angle_{3.14159265358979323846}; // phase_jump: jump angle (rad)
    options::Double freq_step_size_{1.0};        // freq_step: frequency offset while toggled (Hz)
    options::Double chirp_slope_{0.0};           // chirp: linear frequency slope (Hz/s)

    // Synthetic-signal imperfections (apply to every mode)
    options::String noise_color_{"white"};       // white | pink (Voss-McCartney 1/f)
    options::Double snr_db_{20.0};               // in-band SNR (dB): carrier power vs noise
                                                 // power within noise_band_hz around the carrier
    options::Double noise_band_hz_{4.0};         // width of the SNR reference band around carrier_frequency (Hz)
    options::Int noise_seed_{-1};                // additive-noise RNG seed; -1 = seed from std::random_device (non-deterministic)
    options::Double burst_on_s_{1.0};            // bursty signal: ON duration (s)
    options::Double burst_off_s_{0.0};           // bursty signal: OFF duration (s); 0 = always on

    options::Value<unsigned int, false> nchannels_{10};
    options::Value<unsigned int, false> nsamples_{1};
    options::Int n_messages_{-1};

    // Pacing: when true, sleep between packets so output is emitted at real-time rate.
    // When false, packets are generated as fast as possible (no inter-packet sleep).
    options::Bool inter_packet_sleep_{true};

    // Runtime state
    BroadcasterState<double> *f0_state_ = nullptr;
    double current_f0_ = 10.0;
    int packet_count_ = 0;
    TimePoint last_emit_time_{};  // wall-clock time of the previous emit, used to pace output
};
