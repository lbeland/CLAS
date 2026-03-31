#include <sdft/sdft.h>

#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <vector>

using sdft::SDFT;
using sdft::Window;

namespace
{

const double PI = std::acos(-1.0);

double mean_abs(const std::vector<double>& v)
{
  if (v.empty())
  {
    return 0.0;
  }

  double acc = 0.0;

  for (const auto value : v)
  {
    acc += std::abs(value);
  }

  return acc / static_cast<double>(v.size());
}

double max_abs(const std::vector<double>& v)
{
  double m = 0.0;

  for (const auto value : v)
  {
    m = std::max(m, std::abs(value));
  }

  return m;
}

double stdev(const std::vector<double>& v)
{
  if (v.empty())
  {
    return 0.0;
  }

  const double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());

  double var = 0.0;

  for (const auto value : v)
  {
    const double d = value - mean;
    var += d * d;
  }

  var /= static_cast<double>(v.size());

  return std::sqrt(var);
}

} // namespace

int main()
{
  const int signal_length = 2100;
  const int window_length = 1000;
  const int fs = 10000;
  const int n_iterations = signal_length - window_length;

  std::vector<double> x(signal_length, 0.0);

  for (int i = 0; i < signal_length; ++i)
  {
    const double t = static_cast<double>(i) / static_cast<double>(fs);
    x[i] = std::sin(2.0 * PI * 10.0 * t);
  }

  const size_t m = static_cast<size_t>(window_length / 2);
  SDFT<double, double> sdft(m, Window::Boxcar, 1.0);

  const size_t L = static_cast<size_t>(window_length);
  const auto first_begin = x.begin();
  const auto first_end = x.begin() + static_cast<std::ptrdiff_t>(window_length);
  std::vector<double> first_window(first_begin, first_end);

  std::vector<std::complex<double>> X_sdft_matrix(L * m);
  sdft.sdft(L, first_window.data(), X_sdft_matrix.data());

  std::vector<std::complex<double>> X_sdft(m);
  std::copy(X_sdft_matrix.end() - static_cast<std::ptrdiff_t>(m), X_sdft_matrix.end(), X_sdft.begin());

  std::cout << "Initial spectrum shape: (" << window_length << ", " << m << ")" << std::endl;

  std::vector<double> y_sdfts;
  std::vector<double> y_fulls;

  y_sdfts.reserve(static_cast<size_t>(n_iterations));
  y_fulls.reserve(static_cast<size_t>(n_iterations));

  const auto t0_sdft = std::chrono::high_resolution_clock::now();

  for (int start = 1; start < signal_length - window_length; ++start)
  {
    const double new_sample = x[static_cast<size_t>(start + window_length - 1)];

    sdft.sdft(new_sample, X_sdft.data());
    const double y_sdft = sdft.isdft(X_sdft.data());

    y_sdfts.push_back(y_sdft);
  }

  const auto t1_sdft = std::chrono::high_resolution_clock::now();
  const auto sdft_ms = std::chrono::duration<double, std::milli>(t1_sdft - t0_sdft).count() / static_cast<double>(n_iterations);

  std::cout << "SDFT processing time per iteration: " << sdft_ms << " ms" << std::endl;

  std::vector<double> fft_in(static_cast<size_t>(window_length), 0.0);
  std::vector<fftw_complex> fft_bins(static_cast<size_t>(window_length / 2 + 1));
  std::vector<double> ifft_out(static_cast<size_t>(window_length), 0.0);

  fftw_plan plan_fwd = fftw_plan_dft_r2c_1d(window_length, fft_in.data(), fft_bins.data(), FFTW_ESTIMATE);
  fftw_plan plan_inv = fftw_plan_dft_c2r_1d(window_length, fft_bins.data(), ifft_out.data(), FFTW_ESTIMATE);

  if (plan_fwd == nullptr || plan_inv == nullptr)
  {
    std::cerr << "Failed to create FFTW plans." << std::endl;
    if (plan_fwd != nullptr)
    {
      fftw_destroy_plan(plan_fwd);
    }
    if (plan_inv != nullptr)
    {
      fftw_destroy_plan(plan_inv);
    }
    return 1;
  }

  const auto t0_fft = std::chrono::high_resolution_clock::now();

  for (int start = 1; start < signal_length - window_length; ++start)
  {
    const auto wb = x.begin() + start;
    std::copy(wb, wb + window_length, fft_in.begin());

    fftw_execute(plan_fwd);
    fftw_execute(plan_inv);

    const double y_full = ifft_out[static_cast<size_t>(window_length - 1)] / static_cast<double>(window_length);

    y_fulls.push_back(y_full);
  }

  const auto t1_fft = std::chrono::high_resolution_clock::now();
  const auto fft_ms = std::chrono::duration<double, std::milli>(t1_fft - t0_fft).count() / static_cast<double>(n_iterations);

  fftw_destroy_plan(plan_fwd);
  fftw_destroy_plan(plan_inv);

  std::cout << "Full FFT processing time per iteration: " << fft_ms << " ms" << std::endl;

  std::vector<double> errors(y_sdfts.size(), 0.0);

  for (size_t i = 0; i < errors.size(); ++i)
  {
    errors[i] = y_sdfts[i] - y_fulls[i];
  }

  std::cout << "Sliding updates performed: " << n_iterations << std::endl;
  std::cout << "Mean absolute difference between SDFT and full DFT outputs: " << mean_abs(errors) << std::endl;
  std::cout << "Max absolute difference between SDFT and full DFT outputs: " << max_abs(errors) << std::endl;
  std::cout << "Standard deviation of difference between SDFT and full DFT outputs: " << stdev(errors) << std::endl;

  return 0;
}
