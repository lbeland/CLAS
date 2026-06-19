#include <array>
#include <bit>
#include <cstdint>
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
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <iomanip>

constexpr std::size_t FRAME_SIZE = 172;
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

bool parse_frame(const std::uint8_t* buf, std::size_t len, Frame& out) {
    if (len != FRAME_SIZE) {
        return false;
    }

    out.token = read_u32_le(buf + 0);
    out.sample_counter = read_u32_le(buf + 4);
    out.trigger_bits = read_u32_le(buf + 8);

    for (std::size_t i = 0; i < NUM_AUX; ++i) {
        out.aux[i] = read_f32_le(buf + 12 + 4 * i);
    }

    for (std::size_t i = 0; i < NUM_EEG; ++i) {
        out.eeg[i] = read_f32_le(buf + 44 + 4 * i);
    }

    return true;
}

int main() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return 1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(25000);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sockfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        close(sockfd);
        return 1;
    }

    // No connect to specific sender because IP is broadcast

    std::cout << "Listening on UDP port 25000 \n";

    std::array<std::uint8_t, 2048> buffer{};

    const double freq = 500.0; // Hz, nominal device sample rate

    const int n_packets = static_cast<int>(10 * freq); // ~10 seconds worth of packets

    // Use vectors (not a large stack array) and only ever access [0, count)
    std::vector<std::chrono::steady_clock::time_point> receive_times;
    std::vector<uint32_t> sample_counters;
    receive_times.reserve(n_packets);
    sample_counters.reserve(n_packets);

    int count = 0;
    ssize_t len = 0;
    sockaddr_in sender{};
    socklen_t sender_len = sizeof(sender);

    uint32_t last_counter = 0;
    bool have_last_counter = false;
    uint64_t missed_total = 0;
    uint64_t max_gap = 0;

    while (count < n_packets) {

        len = recvfrom(sockfd, buffer.data(), buffer.size(), 0,
                        reinterpret_cast<sockaddr*>(&sender), &sender_len);
        auto timestamp = std::chrono::steady_clock::now();

        if (len < 0) {
            perror("recvfrom");
            break;
        }

        Frame frame{};
        if (!parse_frame(buffer.data(), static_cast<std::size_t>(len), frame)) {
            std::cerr << "Unexpected packet size: " << len << " bytes\n";
            continue;
        }

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

        receive_times.push_back(timestamp);
        sample_counters.push_back(frame.sample_counter);
        count++;
    }

    std::cout << "\nReceived " << count << " packets.\n";
    std::cout << "Missed (interpolated) packets: " << missed_total
              << " (max single gap: " << max_gap << ")\n";
    close(sockfd);

    if (count < 2) {
        std::cerr << "Not enough packets received for analysis.\n";
        return 1;
    }

    // ------------------------------------------------------------------
    // Legacy metric: raw inter-arrival periods.
    // Kept for comparison, but note this measures *changes* in jitter
    // between consecutive packets, not absolute per-packet jitter.
    // ------------------------------------------------------------------
    double sum_diff = 0.0;
    double max_diff = 0.0;
    std::size_t max_idx = 0;
    double sum_sq_diff = 0.0;
    const std::size_t n = receive_times.size() - 1;
    std::vector<double> inter_sample_us(n);

    for (std::size_t i = 0; i < n; i++) {
        double diff = std::chrono::duration_cast<std::chrono::nanoseconds>(
            receive_times[i + 1] - receive_times[i]).count();
        inter_sample_us[i] = diff * 1e-3; // convert to microseconds
        sum_diff += diff;
        sum_sq_diff += diff * diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    double avg_period_ns = sum_diff / static_cast<double>(n);
    double variance_ns2 = (sum_sq_diff / static_cast<double>(n)) - (avg_period_ns * avg_period_ns);
    double std_period_ns = std::sqrt(std::fmax(0.0, variance_ns2));

    std::cout << "\n--- Inter-arrival period (legacy) ---";
    printf("\n Average receive period (us): %.3f", avg_period_ns * 1e-3);
    printf("\n Max receive period (us): %.3f, idx: %zu", max_diff * 1e-3, max_idx);
    printf("\n Std receive period (us): %.3f\n", std_period_ns * 1e-3);

    // ------------------------------------------------------------------
    // Grid-based jitter analysis.
    //
    // offset_i = (a_i - a_0) - (sample_counter_i - sample_counter_0) / fs
    //
    // offset_i drifts linearly if the true device sample rate differs from
    // the nominal `freq`, so we first detrend via linear regression against
    // sample index, then take jitter_i = residual_i - min(residual).
    // jitter_i is now a true non-negative per-packet delay relative to the
    // best-case (lowest-latency) packet observed.
    // ------------------------------------------------------------------
    {
        const std::size_t n = receive_times.size();

        std::vector<double> x(n);       // sample index relative to first packet
        std::vector<double> offset(n);  // offset from ideal grid, in microseconds

        const uint32_t sc0 = sample_counters[0];
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = static_cast<double>(sample_counters[i]) - static_cast<double>(sc0);

            double a_us = std::chrono::duration<double, std::micro>(
                receive_times[i] - receive_times[0]).count();
            double ideal_us = x[i] / freq * 1e6;
            offset[i] = a_us - ideal_us;
        }

        // Linear regression (centered) to estimate drift slope b (us per sample)
        double mean_x = 0.0, mean_offset = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            mean_x += x[i];
            mean_offset += offset[i];
        }
        mean_x /= static_cast<double>(n);
        mean_offset /= static_cast<double>(n);

        double sxy = 0.0, sxx = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double dx = x[i] - mean_x;
            double doff = offset[i] - mean_offset;
            sxy += dx * doff;
            sxx += dx * dx;
        }
        double b_us_per_sample = (sxx > 0.0) ? (sxy / sxx) : 0.0;

        // Detrend
        std::vector<double> residual(n);
        for (std::size_t i = 0; i < n; ++i) {
            double fitted = mean_offset + b_us_per_sample * (x[i] - mean_x);
            residual[i] = offset[i] - fitted;
        }

        double min_residual = *std::min_element(residual.begin(), residual.end());

        std::vector<double> jitter(n);
        for (std::size_t i = 0; i < n; ++i) {
            jitter[i] = residual[i] - min_residual;
        }

        // Stats on jitter
        double sum_j = 0.0, sum_sq_j = 0.0, max_j = 0.0;
        std::size_t max_j_idx = 0;
        for (std::size_t i = 0; i < n; ++i) {
            sum_j += jitter[i];
            sum_sq_j += jitter[i] * jitter[i];
            if (jitter[i] > max_j) {
                max_j = jitter[i];
                max_j_idx = i;
            }
        }
        double mean_j = sum_j / static_cast<double>(n);
        double var_j = (sum_sq_j / static_cast<double>(n)) - (mean_j * mean_j);
        double std_j = std::sqrt(std::fmax(0.0, var_j));

        // Percentiles
        std::vector<double> sorted_jitter = jitter;
        std::sort(sorted_jitter.begin(), sorted_jitter.end());
        auto percentile = [&](double p) {
            std::size_t idx = static_cast<std::size_t>(p * (sorted_jitter.size() - 1));
            return sorted_jitter[idx];
        };

        // Estimated true sample rate from regression slope:
        // d(offset)/dx = 1/true_fs - 1/freq  =>  1/true_fs = 1/freq + b
        double true_fs = freq / (1.0 + freq * b_us_per_sample * 1e-6);

        std::cout << "\n--- Grid-based jitter (relative to sample_counter) ---";
        printf("\n Estimated true sample rate (Hz): %.4f (nominal %.1f)", true_fs, freq);
        printf("\n Drift slope (us per sample):     %.6f", b_us_per_sample);
        printf("\n Jitter mean (us):  %.3f", mean_j);
        printf("\n Jitter std  (us):  %.3f", std_j);
        printf("\n Jitter min  (us):  %.3f (by construction, = 0)", *std::min_element(jitter.begin(), jitter.end()));
        printf("\n Jitter max  (us):  %.3f, idx: %zu", max_j, max_j_idx);
        printf("\n Jitter p50  (us):  %.3f", percentile(0.50));
        printf("\n Jitter p95  (us):  %.3f", percentile(0.95));
        printf("\n Jitter p99  (us):  %.3f\n", percentile(0.99));

        // Save per-packet jitter for further analysis / plotting
        std::ofstream out("jitter_" + std::to_string((int)freq) + ".csv");
        if (!out.is_open()) {
            std::cerr << "Failed to open jitter_" << (int)freq << ".csv for writing\n";
            return 1;
        }
        out << "sample_counter,jitter_us,inter_sample_us\n";
        for (std::size_t i = 0; i < n; ++i) {
            out << sample_counters[i] << "," << jitter[i] << "," << inter_sample_us[i] << "\n";
        }
        out.close();
        std::cout << "Per-packet jitter saved to jitter_" << (int)freq << ".csv\n";
    }

    return 0;
}