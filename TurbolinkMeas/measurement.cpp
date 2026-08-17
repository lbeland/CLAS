#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <arpa/inet.h>
#include <chrono>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <limits>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <iomanip>

constexpr std::size_t FRAME_SIZE = 172;
constexpr std::size_t NUM_PACKAGING = 1;
constexpr std::size_t NUM_AUX = 8;
constexpr std::size_t NUM_EEG = 32;

// P(n0) = C * I: initial RLS covariance scale, i.e. a large, poorly
// confident prior around the nominal-rate initial guess. [Ikonen 2002]
constexpr double CLOCK_RLS_INIT_C = 50.0;

struct Frame {
    uint32_t token;
    uint32_t sample_counter;
    uint32_t trigger_bits;
    std::array<float, NUM_AUX> aux;
    std::array<float, NUM_EEG> eeg;
};

uint32_t read_u32_le(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

float bits_to_float(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

float read_f32_le(const std::uint8_t* p) {
    static_assert(sizeof(float) == 4);
    uint32_t bits;
    std::memcpy(&bits, p, sizeof(bits));

    return bits_to_float(bits);
}

bool parse_packet(const std::uint8_t* buf, std::size_t len, std::array<Frame, NUM_PACKAGING>& out) {
    if (len != FRAME_SIZE * NUM_PACKAGING) {
        return false;
    }

    for (std::size_t p = 0; p < NUM_PACKAGING; ++p) {
        const std::uint8_t* frame_buf = buf + p * FRAME_SIZE;
        Frame& out_frame = out[p];

        out_frame.token = read_u32_le(frame_buf + 0);
        out_frame.sample_counter = read_u32_le(frame_buf + 4);
        out_frame.trigger_bits = read_u32_le(frame_buf + 8);

        for (std::size_t i = 0; i < NUM_AUX; ++i) {
            out_frame.aux[i] = read_f32_le(frame_buf + 12 + 4 * i);
        }

        for (std::size_t i = 0; i < NUM_EEG; ++i) {
            out_frame.eeg[i] = read_f32_le(frame_buf + 44 + 4 * i);
        }
    }

    return true;
}

static double diff_ns(const timespec& start, const timespec& end) {
    double sec_diff = static_cast<double>(end.tv_sec - start.tv_sec);
    double nsec_diff = static_cast<double>(end.tv_nsec - start.tv_nsec);
    return sec_diff * 1e9 + nsec_diff;
}

// Online (O(1)-memory) running statistics, so this program can run for
// arbitrarily long sessions (e.g. 30 min) without accumulating one entry
// per sample/packet in memory.

// Welford's online mean/variance, plus a running max (with its index).
struct RunningStats {
    uint64_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double max_val = -std::numeric_limits<double>::infinity();
    uint64_t max_idx = 0;

    void update(double x, uint64_t idx) {
        ++count;
        double delta = x - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (x - mean);
        if (x > max_val) {
            max_val = x;
            max_idx = idx;
        }
    }

    // Population variance (matches the original sum_sq/n - mean^2 formula).
    double variance() const {
        return count > 0 ? m2 / static_cast<double>(count) : 0.0;
    }
    double stddev() const { return std::sqrt(std::fmax(0.0, variance())); }
};

// Recursive Least Squares (RLS) estimator for the ActiCHamp clock model
//
//   y(n) = theta^T phi(n) + xi(n),   y(n) = t(n) (measured timestamp),
//   theta = [theta1, theta2]^T = [1/fs, t0]^T,
//   phi(n) = [phi1(n), phi2(n)]^T = [n - n0, 1]^T,
//
// i.e. the true sampling period theta1 AND the start offset theta2 = t0
// are both estimated (unlike a slope-only fit anchored to the first
// sample). Updated in O(1) per sample:
//
//   L(n)      = P(n-1) phi(n) / (1/alpha + phi^T(n) P(n-1) phi(n))
//   theta(n)  = theta(n-1) + L(n) [y(n) - theta^T(n-1) phi(n)]
//   P(n)      = P(n-1) - L(n) phi^T(n) P(n-1)
//
// alpha is the forgetting factor (alpha=1: every sample weighted equally,
// appropriate for these short, stationary-rate runs). [Ikonen 2002]
struct ClockRLS {
    // theta(n0) = [1/fs_nom, t_init]: the true rate is assumed close to the
    // nominal rate, and the start offset close to the absolute timestamp
    // t_init recorded just before the measurement starts (both set by the
    // caller before the first update() call).
    double theta1 = 0.0; // estimated sampling period, us/sample (1/fs)
    double theta2 = 0.0; // estimated start offset t0, us

    // P(n0) = C * I (2x2, symmetric: P12 == P21): a large C means a poorly
    // confident prior, so early updates converge onto the batch
    // least-squares solution. Set by the caller before the first update().
    double P11 = 0.0, P12 = 0.0, P22 = 0.0;

    double alpha = 1.0; // forgetting factor (alpha_n = 1: equal weighting of all samples)

    void update(double phi1, double phi2, double y) {
        // P(n-1) * phi(n)
        const double Pphi1 = P11 * phi1 + P12 * phi2;
        const double Pphi2 = P12 * phi1 + P22 * phi2;

        // L(n) = P(n-1) phi(n) / (1/alpha + phi^T(n) P(n-1) phi(n))
        const double denom = 1.0 / alpha + phi1 * Pphi1 + phi2 * Pphi2;
        const double L1 = Pphi1 / denom;
        const double L2 = Pphi2 / denom;

        // theta(n) = theta(n-1) + L(n) [y(n) - theta^T(n-1) phi(n)]
        const double error = y - (theta1 * phi1 + theta2 * phi2);
        theta1 += L1 * error;
        theta2 += L2 * error;

        // P(n) = P(n-1) - L(n) phi^T(n) P(n-1)
        //      = P(n-1) - L(n) (P(n-1) phi(n))^T   (P symmetric)
        P11 -= L1 * Pphi1;
        P12 -= L1 * Pphi2; // == P22-side term by symmetry
        P22 -= L2 * Pphi2;
    }

    // Fitted y at regressor phi = [phi1, phi2].
    double fitted(double phi1, double phi2) const { return theta1 * phi1 + theta2 * phi2; }
};

// One packet's grid-PDV (Packet Delay Variation) inputs, kept only for the
// most recent PDV_WINDOW_S seconds (see PdvRingBuffer) so we can still
// export a CSV for plotting without retaining the full-run history.
struct PdvSample {
    uint32_t sample_counter = 0;
    double x = 0.0;      // phi1(n) = n - n0, sample index relative to the first packet
    double t_us = 0.0;   // y(n): measured arrival timestamp (us, relative to start_time)
    double inter_sample_us = std::numeric_limits<double>::quiet_NaN(); // gap to the *next* packet; backfilled when it arrives
};

// Fixed-capacity circular buffer: memory is bounded by the window length,
// not by the run length.
class PdvRingBuffer {
public:
    explicit PdvRingBuffer(std::size_t capacity) : buf_(capacity) {}

    void push(const PdvSample& sample) {
        buf_[next_write_] = sample;
        last_write_idx_ = next_write_;
        next_write_ = (next_write_ + 1) % buf_.size();
        if (filled_ < buf_.size()) ++filled_;
    }

    // Sets inter_sample_us on the most recently pushed entry.
    void backfill_inter_sample_us(double value) {
        if (last_write_idx_ != SIZE_MAX) {
            buf_[last_write_idx_].inter_sample_us = value;
        }
    }

    // Returns buffered samples in chronological order (oldest first).
    std::vector<PdvSample> ordered() const {
        std::vector<PdvSample> out;
        out.reserve(filled_);
        if (filled_ < buf_.size()) {
            out.insert(out.end(), buf_.begin(), buf_.begin() + static_cast<long>(filled_));
        } else {
            out.insert(out.end(), buf_.begin() + static_cast<long>(next_write_), buf_.end());
            out.insert(out.end(), buf_.begin(), buf_.begin() + static_cast<long>(next_write_));
        }
        return out;
    }

private:
    std::vector<PdvSample> buf_;
    std::size_t next_write_ = 0;
    std::size_t filled_ = 0;
    std::size_t last_write_idx_ = SIZE_MAX;
};

int main(int argc, char* argv[]) {
    double freq = 10000.0; // Hz, nominal device sample rate
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [nominal_sample_rate_hz]\n";
        return 1;
    }
    if (argc == 2) {
        char* end = nullptr;
        double parsed = std::strtod(argv[1], &end);
        if (end == argv[1] || *end != '\0' || !(parsed > 0.0)) {
            std::cerr << "Invalid nominal_sample_rate_hz '" << argv[1] << "', falling back to " << freq << " Hz\n";
        } else {
            freq = parsed;
        }
    }
    std::cout << "Nominal sample rate: " << freq << " Hz\n";

    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return 1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(25000);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    static_assert(FRAME_SIZE * NUM_PACKAGING <= 2048, "buffer too small for NUM_PACKAGING");
    std::array<std::uint8_t, 2048> buffer{};

    const int n_samples = static_cast<int>(2 * 60 * freq);

    // Tail window kept for CSV/percentile output
    constexpr double PDV_WINDOW_S = 10.0;
    const std::size_t ring_capacity =
        static_cast<std::size_t>(PDV_WINDOW_S * freq / static_cast<double>(NUM_PACKAGING)) + 1;
    PdvRingBuffer ring(ring_capacity);

    int count = 0; // number of individual samples decoded (NUM_PACKAGING per network packet)
    int net_packet_count = 0; // number of UDP packets received
    ssize_t len = 0;
    sockaddr_in sender{};
    socklen_t sender_len = sizeof(sender);

    uint32_t last_counter = 0;
    bool have_last_counter = false;
    uint64_t missed_total = 0;
    uint64_t max_gap = 0;

    // Legacy inter-arrival period stats (packet-to-packet), O(1) memory.
    RunningStats legacy_period_stats;
    uint64_t diff_idx = 0;

    uint32_t sc0 = 0;
    bool have_first_packet = false;
    std::chrono::steady_clock::time_point prev_timestamp{};
    bool have_prev_timestamp = false;

    // No connect to specific sender because IP is broadcast
    if (bind(sockfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        close(sockfd);
        return 1;
    }

    // t_init: absolute timestamp recorded just before the measurement
    // starts. Also serves as the epoch for y(n): all measured timestamps
    // below are expressed as microseconds elapsed since start_time, so in
    // that frame t_init == 0 and theta2's initial guess is 0 accordingly.
    // start_time_wall is a wall-clock anchor taken alongside start_time,
    // used only to report theta2 as an absolute timestamp (steady_clock's
    // own epoch is unspecified/not human-readable).
    auto start_time = std::chrono::steady_clock::now();
    auto start_time_wall = std::chrono::system_clock::now();

    // Clock RLS (see analysis below), O(1) memory: theta(n0) = [1/fs_nom,
    // t_init], P(n0) = C * I.
    ClockRLS clock_rls;
    clock_rls.theta1 = 1e6 / freq;
    clock_rls.theta2 = 0.0;
    clock_rls.P11 = CLOCK_RLS_INIT_C;
    clock_rls.P22 = CLOCK_RLS_INIT_C;

    while (count < n_samples) {

        len = recvfrom(sockfd, buffer.data(), buffer.size(), 0,
                        reinterpret_cast<sockaddr*>(&sender), &sender_len);
        auto timestamp = std::chrono::steady_clock::now();

        if (len < 0) {
            perror("recvfrom");
            break;
        }

        std::array<Frame, NUM_PACKAGING> frames{};
        if (!parse_packet(buffer.data(), static_cast<std::size_t>(len), frames)) {
            std::cerr << "Unexpected packet size: " << len << " bytes\n";
            continue;
        }
        net_packet_count++;

        // A single UDP packet carries NUM_PACKAGING samples back-to-back.
        // Check every sample's counter for continuity (a gap can occur at any
        // sample boundary, not just packet boundaries), but only log one
        // (timestamp, sample_counter) pair for the whole packet below.
        for (const Frame& frame : frames) {
            if (!have_last_counter) {
                last_counter = frame.sample_counter - 1;
                have_last_counter = true;
            }

            uint32_t expected = last_counter + 1;
            if (frame.sample_counter != expected) {
                uint64_t gap = static_cast<uint64_t>(frame.sample_counter) - static_cast<uint64_t>(expected) + 1;
                std::cerr << "Warning: Missed packet(s). Last counter: " << last_counter
                          << ", current: " << frame.sample_counter
                          << " (" << gap << " missed)\n";
                missed_total += gap;
                max_gap = std::max(max_gap, gap);
            }
            last_counter = frame.sample_counter;
            count++;
        }

        // Tag the packet with the last sample's counter: that's the most
        // recently captured sample as of this packet's arrival.
        const uint32_t sc = frames.back().sample_counter;

        // Legacy inter-arrival period, folded into running stats instead of
        // a stored vector. Also backfills the previous ring-buffer entry's
        // inter_sample_us now that we know the gap to this packet.
        if (have_prev_timestamp) {
            double diff = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                timestamp - prev_timestamp).count());
            legacy_period_stats.update(diff, diff_idx++);
            ring.backfill_inter_sample_us(diff * 1e-3); // ns -> us
        }
        prev_timestamp = timestamp;
        have_prev_timestamp = true;

        // Clock RLS input for this packet: phi(n) = [n - n0, 1], y(n) = t(n).
        if (!have_first_packet) {
            sc0 = sc;
            have_first_packet = true;
        }
        double x = static_cast<double>(sc) - static_cast<double>(sc0);
        double t_us = std::chrono::duration<double, std::micro>(timestamp - start_time).count();
        clock_rls.update(x, 1.0, t_us);

        PdvSample ps;
        ps.sample_counter = sc;
        ps.x = x;
        ps.t_us = t_us;
        ring.push(ps);
        // auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - timestamp).count();
        // if (elapsed > 10) {
        //     std::cerr << "Warning: Processing took" << elapsed << " microseconds\n";
        // }
    }
    auto end_time = std::chrono::steady_clock::now();

    std::cout << "\nReceived " << count << " samples ("
    << net_packet_count << " UDP packets, " << NUM_PACKAGING << " samples/packet) in "
    << std::fixed << std::setprecision(3)
    << static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / 1000.0
    << "ms.\n"
    << "Estimated fs: " << std::fixed << std::setprecision(4) << static_cast<double>(count) / std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count()
    << "\n";

    std::cout << "Missed packets: " << missed_total << ")\n";
    close(sockfd);

    if (net_packet_count < 2) {
        std::cerr << "Not enough packets received for analysis.\n";
        return 1;
    }

    // Legacy metric: raw inter-arrival periods
    printf("\n--- Inter-arrival period (legacy) ---");
    printf("\n Average receive period (us): %.5f", legacy_period_stats.mean * 1e-3);
    printf("\n Estimated packet rate: %.4f Hz (%.4f Hz at %zu samples/packet)",
           1e9 / legacy_period_stats.mean, (1e9 / legacy_period_stats.mean) * NUM_PACKAGING, NUM_PACKAGING);
    printf("\n Max receive period (us): %.3f, idx: %llu", legacy_period_stats.max_val * 1e-3,
           static_cast<unsigned long long>(legacy_period_stats.max_idx));
    printf("\n Std receive period (us): %.3f\n", legacy_period_stats.stddev() * 1e-3);

    // Clock RLS / PDV (Packet Delay Variation) analysis.
    //
    // t(n) = theta1 * (n - n0) + theta2 + xi(n), fitted online in O(1) via
    // recursive least squares (clock_rls above, alpha_n = 1) rather than a
    // two-pass batch fit over stored vectors. residual_i = t_i - fitted(x_i)
    // is the PDV before flooring, pdv_i = residual_i - min(residual).
    //
    // pdv_i requires knowing min(residual) over the run, which can't be
    // obtained exactly in O(1) memory from an unbounded stream; per-sample
    // PDV/percentiles below are therefore computed only over the most
    // recent PDV_WINDOW_S seconds (PdvRingBuffer), not the full run.
    {
        // theta1 = estimated true sampling period (us/sample); invert for Hz.
        const double true_fs = 1e6 / clock_rls.theta1;
        const double drift_ppm = (true_fs - freq) / freq * 1e6;

        std::vector<PdvSample> tail = ring.ordered();

        std::vector<double> residual(tail.size());
        for (std::size_t i = 0; i < tail.size(); ++i) {
            residual[i] = tail[i].t_us - clock_rls.fitted(tail[i].x, 1.0);
        }

        double min_residual = tail.empty() ? 0.0 : *std::min_element(residual.begin(), residual.end());

        std::vector<double> pdv(tail.size());
        for (std::size_t i = 0; i < tail.size(); ++i) {
            pdv[i] = residual[i] - min_residual;
        }

        // Absolute t0 (theta2), recovered by anchoring the RLS's start_time-
        // relative offset to the wall-clock time sampled alongside it.
        const auto true_origin = start_time_wall +
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::duration<double, std::micro>(clock_rls.theta2));
        const std::time_t true_origin_tt = std::chrono::system_clock::to_time_t(true_origin);

        std::cout << "\n--- Clock RLS / PDV (relative to sample_counter) ---";
        printf("\n Estimated true sample rate (Hz): %.4f (nominal %.1f)", true_fs, freq);
        printf("\n Estimated sampling period (us):  %.6f", clock_rls.theta1);
        printf("\n Estimated start offset t0 (us):  %.6f", clock_rls.theta2);
        std::cout << "\n Estimated true clock origin t0:  "
                   << std::put_time(std::localtime(&true_origin_tt), "%Y-%m-%d %H:%M:%S");
        printf(".%06lld", static_cast<long long>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                true_origin - std::chrono::system_clock::from_time_t(true_origin_tt)).count()));
        printf("\n Clock drift: %.4f ppm\n", drift_ppm);

        // Save per-packet PDV for the buffered tail window (not the full
        // run) for further analysis / plotting in plotMeas.py
        std::ofstream out("pdv_" + std::to_string((int)freq) + ".csv");
        if (!out.is_open()) {
            std::cerr << "Failed to open pdv_" << (int)freq << ".csv for writing\n";
            return 1;
        }
        out << "# true_fs_hz=" << std::fixed << std::setprecision(6) << true_fs << "\n";
        out << "sample_counter,pdv_us,inter_sample_us\n";
        for (std::size_t i = 0; i < tail.size(); ++i) {
            // The most recently buffered sample has no "gap to next" yet.
            out << tail[i].sample_counter << "," << pdv[i] << ",";
            if (!std::isnan(tail[i].inter_sample_us)) {
                out << tail[i].inter_sample_us;
            }
            out << "\n";
        }
        out.close();
        std::cout << "Per-packet PDV (last " << PDV_WINDOW_S << "s) saved to pdv_" << (int)freq << ".csv\n";
    }

    return 0;
}
