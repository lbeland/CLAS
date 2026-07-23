// Compare Duda's modulated Sliding DFT (mSDFT) against a per-sample
// recomputed windowed FFT using FFTW: processing time per iteration,
// agreement of the estimated magnitude spectra, and whether that error
// drifts/accumulates over a long run (the sdft recursion carries its state
// forward forever, unlike the FFT which recomputes from scratch every
// iteration).
//
// Writes the per-iteration error to sdft_error_growth.bin; plot with
// plot_error_growth.py to see it over time.
//
// Build:
//   g++ -O3 -march=native -std=c++17 sdft_comparison.cpp -lfftw3 -o bench
// Run:
//   ./bench

#include "sdft/cpp/src/sdft/sdft.h"

#include <fftw3.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <vector>

using clk = std::chrono::high_resolution_clock;

static double us_since(const clk::time_point& t0, const clk::time_point& t1)
{
  return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

int main()
{
  // ---- signal setup ----
  // n_iterations is large enough to also see whether the error drifts over
  // a long run, not just its magnitude on a short one.
  const size_t n_iterations  = 100; // 5 min at 10 kHz
  const size_t window_length = 100000;  // 10 s at 10 kHz
  const size_t signal_length = n_iterations + window_length;

  const double fs   = 10000.0;
  const double freq = 6.35672;

  std::vector<double> x(signal_length);
  for (size_t i = 0; i < signal_length; ++i)
  {
    const double t = static_cast<double>(i) / fs;
    x[i] = std::sin(2.0 * M_PI * freq * t);
  }

  const size_t m = window_length / 2;  // DFT size (bins), same convention as sdft() half-size output

  // ================= mSDFT (jurihock/sdft, boxcar window, latency=1) =================
  sdft::SDFT<double, double> sdft(m, sdft::Window::Boxcar, 1.0);

  // Warm up / prime with the first full window
  std::vector<std::complex<double>> dft(m);
  for (size_t i = 0; i < window_length; ++i)
    sdft.sdft(x[i], dft.data());

  // ================= Sliding full FFT with FFTW (recomputed each shift) =================
  // FFTW_PATIENT spends noticeably longer at plan-creation time in exchange
  // for the fastest codelets it can find for this exact transform size on
  // this machine; since the plan is built once and reused for tens of
  // thousands of iterations, that cost is amortized to ~nothing.
  std::vector<double>               win_buf(window_length);
  std::vector<std::complex<double>> full_out(window_length / 2 + 1);

  fftw_plan plan_fwd = fftw_plan_dft_r2c_1d(
      static_cast<int>(window_length),
      win_buf.data(),
      reinterpret_cast<fftw_complex*>(full_out.data()),
      FFTW_PATIENT);

  // Both spectra have to be computed for the *same* window before a
  // meaningful per-bin comparison is possible, so this is a single loop.
  // Timing for each method is still tracked separately via its own
  // accumulator so the two don't get conflated.
  double sdft_time_us = 0.0;
  double full_time_us = 0.0;

  // Mean per-bin absolute difference per iteration. abs() is taken bin by
  // bin, before averaging, so a mismatch that lands in different bins
  // (e.g. sdft energy at k=3 vs FFT energy at k=5) can't cancel itself out
  // the way it would if the signed differences were averaged first.
  std::vector<double> errors_freq(n_iterations);

  for (size_t start = 1; start < signal_length - window_length + 1; ++start)
  {
    printf("Processing window starting at index: %zu\n", start);
    const double new_sample = x[start + window_length - 1];

    const auto ta = clk::now();
    sdft.sdft(new_sample, dft.data());
    const auto tb = clk::now();
    sdft_time_us += us_since(ta, tb);

    const auto tc = clk::now();
    std::copy(x.begin() + start, x.begin() + start + window_length, win_buf.begin());
    fftw_execute(plan_fwd);   // win_buf -> full_out (unnormalized rfft)
    const auto td = clk::now();
    full_time_us += us_since(tc, td);

    // X_full[0:-1] / window_length, i.e. drop the Nyquist bin.
    double sum = 0.0;
    for (size_t k = 0; k < m; ++k)
    {
      const double sdft_mag = std::abs(dft[k]);
      const double full_mag = std::abs(full_out[k]) / window_length;
      sum += std::abs(sdft_mag - full_mag);
    }
    errors_freq[start - 1] = sum / m;
  }

  const double sdft_us_per_iter = sdft_time_us / n_iterations;
  const double full_us_per_iter = full_time_us / n_iterations;

  fftw_destroy_plan(plan_fwd);

  auto mean_abs = [](const std::vector<double>& v) {
    double s = 0.0;
    for (double e : v) s += std::abs(e);
    return s / v.size();
  };
  auto max_abs = [](const std::vector<double>& v) {
    double m = 0.0;
    for (double e : v) m = std::max(m, std::abs(e));
    return m;
  };
  auto stddev = [](const std::vector<double>& v) {
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double s = 0.0;
    for (double e : v) s += (e - mean) * (e - mean);
    return std::sqrt(s / v.size());
  };

  std::printf("Iterations performed: %zu\n\n", n_iterations);

  std::printf("mSDFT    processing time per iteration: %.4f us\n", sdft_us_per_iter);
  std::printf("Full FFT processing time per iteration: %.4f us\n", full_us_per_iter);
  std::printf("Speedup (FFT time / mSDFT time): %.2fx\n\n", full_us_per_iter / sdft_us_per_iter);

  std::printf("----------- Freq (per-bin absolute error) -----------\n");
  std::printf("Mean absolute difference: %g\n", mean_abs(errors_freq));
  std::printf("Max absolute difference:  %g\n", max_abs(errors_freq));
  std::printf("Std deviation:            %g\n", stddev(errors_freq));

  // Quick drift check without needing the plot: compare the first half's
  // mean error against the second half's. If the sdft recursion's state
  // were accumulating numerical error over time, the second half should be
  // visibly larger.
  const size_t half = n_iterations / 2;
  double first_half = 0.0, second_half = 0.0;
  for (size_t i = 0; i < half; ++i) first_half += errors_freq[i];
  for (size_t i = half; i < n_iterations; ++i) second_half += errors_freq[i];
  first_half /= half;
  second_half /= (n_iterations - half);

  std::printf("\n----------- Drift over time -----------\n");
  std::printf("Mean error, first half:  %g\n", first_half);
  std::printf("Mean error, second half: %g\n", second_half);
  std::printf("Ratio (second/first):    %g\n", second_half / first_half);

  std::ofstream out("sdft_error_growth.bin", std::ios::binary);
  out.write(reinterpret_cast<const char*>(errors_freq.data()),
            static_cast<std::streamsize>(errors_freq.size() * sizeof(double)));
  std::printf("\nWrote per-iteration error to sdft_error_growth.bin (%zu doubles) "
              "-- plot with plot_error_growth.py\n", errors_freq.size());

  return 0;
}
