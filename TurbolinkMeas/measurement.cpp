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

    // int busy_poll_us = 10;
    // if (setsockopt(sockfd, SOL_SOCKET, SO_BUSY_POLL,
    //             &busy_poll_us, sizeof(busy_poll_us)) < 0) {
    //     perror("setsockopt(SO_BUSY_POLL)");
    // }

    if (bind(sockfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        close(sockfd);
        return 1;
    }

    // No connect to specific sender because IP is broadcast 

    std::cout << "Listening on UDP port 25000 \n";

    std::array<std::uint8_t, 2048> buffer{};

    const int freq = 5000; // Hz, has to be adapted to the actual frequency of the incoming packets

    static const int n_packets = 5 * freq; // 5 seconds worth of packets 
    std::array<std::chrono::_V2::steady_clock::time_point, n_packets> receive_times;
    int count = 0;
    std::chrono::_V2::steady_clock::time_point timestamp;
    ssize_t len = 0;
    sockaddr_in sender{};
    socklen_t sender_len = sizeof(sender);
    int last_counter;

    while (count<n_packets) {

        len = recvfrom(sockfd, buffer.data(), buffer.size(), 0, reinterpret_cast<sockaddr*>(&sender), &sender_len);
        timestamp = std::chrono::steady_clock::now();

        if (len < 0) {
            perror("recvfrom");
            break;
        }
        receive_times[count] = timestamp;


        Frame frame{};
        if (!parse_frame(buffer.data(), static_cast<std::size_t>(len), frame)) {
            std::cerr << "Unexpected packet size: " << len << " bytes\n";
            continue;
        }
        // for (std::size_t i = 0; i < len; ++i) {
        //     if (i % 16 == 0) std::cout << '\n';
        //     std::cout << std::hex << std::setw(2) << std::setfill('0')
        //               << static_cast<unsigned>(buffer.data()[i]) << ' ';
        // }
        // std::cout << std::dec << '\n';
        // std::cout << "Sample counter: " << frame.sample_counter << "\n";
        if (count == 0){
            last_counter = frame.sample_counter-1;
        }

        if (frame.sample_counter != static_cast<uint32_t>(last_counter + 1)) {
            std::cerr << "Warning: Missed packet(s). Last counter: " << last_counter << ", current: " << frame.sample_counter << "\n";
        }
        last_counter = frame.sample_counter;
        count++;
    }

    std::cout << "\nReceived " << count << " packets.\n";
    close(sockfd);


    if (receive_times.empty()) {
        return 1;
    }

    // Calculate statistics
    double sum_diff = 0.0;
    double max_diff = 0.0;
    std::size_t max_idx = 0;
    double sum_sq_diff = 0.0;
    std::vector<double> receive_times_diff;
    receive_times_diff.resize(receive_times.size() - 1);

    for (std::size_t i = 0; i < receive_times.size()-1; i++) {
        double diff = std::chrono::duration_cast<std::chrono::nanoseconds>(receive_times[i+1] - receive_times[i]).count();
        receive_times_diff[i] = diff;
        sum_diff += diff;
        sum_sq_diff += diff * diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    double avg_period_sec = sum_diff / (receive_times.size() - 1);
    double avg_period = avg_period_sec * 1e-3;  // us
    double variance = (sum_sq_diff / (receive_times.size() - 1)) - (avg_period_sec * avg_period_sec);  // s²
    double std_period = sqrt(fmax(0.0, variance)) * 1e-3;  // us

    printf("\n Average receive period (us): %.6f", avg_period);
    printf("\n Max receive period (us): %.6f, idx: %zu", max_diff * 1e-3, max_idx);
    printf("\n Std receive period (us): %.6f\n", std_period);

    std::ofstream receive_times_output;
    receive_times_output.open("receive_times.csv");
    if (!receive_times_output.is_open()) {
        std::cerr << "Failed to open receive_times.csv for writing\n";
        return 1;
    }
    for (double t : receive_times_diff) {
        receive_times_output << t << "\n";
    }
    receive_times_output.close();
    std::cout << "Results saved to receive_times.csv\n";

    return 0;
}