#pragma once

#include <mutex>

namespace dsp::fftw {

// FFTW plan creation/destruction touches global planner state and is not thread-safe.
inline std::mutex planner_mutex;

}  // namespace dsp::fftw