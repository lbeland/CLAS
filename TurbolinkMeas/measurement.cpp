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
// per sample/packet in memory. Mirrors the Welford's-algorithm approach
// used in SourceClient.cpp's recalibrate_fs_().

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

// Welford's online covariance update, used to fit a linear regression
// (x -> y) incrementally with O(1) memory. This is mathematically exact,
// not an approximation of the two-pass batch formula -- it just spreads
// the same computation over one sample at a time.
struct OnlineRegression {
    uint64_t count = 0;
    double mean_x = 0.0;
    double mean_y = 0.0;
    double cov_xy = 0.0;
    double var_x = 0.0;

    void update(double x, double y) {
        ++count;
        double dx = x - mean_x;
        mean_x += dx / static_cast<double>(count);
        mean_y += (y - mean_y) / static_cast<double>(count);
        cov_xy += dx * (y - mean_y);
        var_x += dx * (x - mean_x);
    }

    double slope() const { return var_x > 0.0 ? cov_xy / var_x : 0.0; }
};

// One packet's grid-jitter inputs, kept only for the most recent
// JITTER_WINDOW_S seconds (see JitterRingBuffer) so we can still export a
// CSV for plotting without retaining the full-run history.
struct JitterSample {
    uint32_t sample_counter = 0;
    double x = 0.0;           // sample index relative to the first packet
    double offset_us = 0.0;   // arrival offset from the ideal sample grid (us)
    double inter_sample_us = std::numeric_limits<double>::quiet_NaN(); // gap to the *next* packet; backfilled when it arrives
};

// Fixed-capacity circular buffer: memory is bounded by the window length,
// not by the run length.
class JitterRingBuffer {
public:
    explicit JitterRingBuffer(std::size_t capacity) : buf_(capacity) {}

    void push(const JitterSample& sample) {
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
    std::vector<JitterSample> ordered() const {
        std::vector<JitterSample> out;
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
    std::vector<JitterSample> buf_;
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
        freq = std::strtod(argv[1], &end);
        if (end == argv[1] || *end != '\0' || !(freq > 0.0)) {
            std::cerr << "Usage: " << argv[0] << " [nominal_sample_rate_hz]\n";
            return 1;
        }
    }

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

    const int n_samples = static_cast<int>(5 * 60 * freq);

    // Tail window kept for CSV/percentile output
    constexpr double JITTER_WINDOW_S = 10.0;
    const std::size_t ring_capacity =
        static_cast<std::size_t>(JITTER_WINDOW_S * freq / static_cast<double>(NUM_PACKAGING)) + 1;
    JitterRingBuffer ring(ring_capacity);

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

    // Grid-based jitter regression inputs (see analysis below), O(1) memory.
    OnlineRegression grid_regression;
    std::chrono::steady_clock::time_point t0{};
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

    auto start_time = std::chrono::steady_clock::now();

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

        // Grid-based jitter regression input for this packet
        if (!have_first_packet) {
            t0 = timestamp;
            sc0 = sc;
            have_first_packet = true;
        }
        double x = static_cast<double>(sc) - static_cast<double>(sc0);
        double a_us = std::chrono::duration<double, std::micro>(timestamp - t0).count();
        double ideal_us = x / freq * 1e6;
        double offset_us = a_us - ideal_us;
        grid_regression.update(x, offset_us);

        JitterSample js;
        js.sample_counter = sc;
        js.x = x;
        js.offset_us = offset_us;
        ring.push(js);
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

    // Grid-based jitter analysis.
    //
    // offset_i = (a_i - a_0) - (sample_counter_i - sample_counter_0) / fs
    //
    // offset_i drifts linearly if the true device sample rate differs from
    // the nominal `freq`, so we detrend via linear regression against sample
    // index, then take jitter_i = residual_i - min(residual). The regression
    // itself (true_fs, drift slope) is exact over the whole run, computed
    // online via Welford's covariance update (grid_regression above) rather
    // than a two-pass batch fit over stored vectors.
    //
    // jitter_i requires knowing min(residual) over the run, which can't be
    // obtained exactly in O(1) memory from an unbounded stream; per-sample
    // jitter/percentiles below are therefore computed only over the most
    // recent JITTER_WINDOW_S seconds (JitterRingBuffer), not the full run.
    {
        const double mean_x = grid_regression.mean_x;
        const double mean_offset = grid_regression.mean_y;
        const double b_us_per_sample = grid_regression.slope();

        // Estimated true sample rate from regression slope:
        // d(offset)/dx = 1/true_fs - 1/freq  =>  1/true_fs = 1/freq + b
        const double true_fs = freq / (1.0 + freq * b_us_per_sample * 1e-6);

        std::vector<JitterSample> tail = ring.ordered();

        std::vector<double> residual(tail.size());
        for (std::size_t i = 0; i < tail.size(); ++i) {
            double fitted = mean_offset + b_us_per_sample * (tail[i].x - mean_x);
            residual[i] = tail[i].offset_us - fitted;
        }

        double min_residual = tail.empty() ? 0.0 : *std::min_element(residual.begin(), residual.end());

        std::vector<double> jitter(tail.size());
        for (std::size_t i = 0; i < tail.size(); ++i) {
            jitter[i] = residual[i] - min_residual;
        }

        std::cout << "\n--- Grid-based jitter (relative to sample_counter) ---";
        printf("\n Estimated true sample rate (Hz): %.4f (nominal %.1f)", true_fs, freq);
        printf("\n Drift slope (us per sample):     %.6f\n", b_us_per_sample);

        // Save per-packet jitter for the buffered tail window (not the full
        // run) for further analysis / plotting in plotMeas.py
        std::ofstream out("jitter_" + std::to_string((int)freq) + ".csv");
        if (!out.is_open()) {
            std::cerr << "Failed to open jitter_" << (int)freq << ".csv for writing\n";
            return 1;
        }
        out << "# true_fs_hz=" << std::fixed << std::setprecision(6) << true_fs << "\n";
        out << "sample_counter,jitter_us,inter_sample_us\n";
        for (std::size_t i = 0; i < tail.size(); ++i) {
            // The most recently buffered sample has no "gap to next" yet.
            out << tail[i].sample_counter << "," << jitter[i] << ",";
            if (!std::isnan(tail[i].inter_sample_us)) {
                out << tail[i].inter_sample_us;
            }
            out << "\n";
        }
        out.close();
        std::cout << "Per-packet jitter (last " << JITTER_WINDOW_S << "s) saved to jitter_" << (int)freq << ".csv\n";
    }

    return 0;
}
