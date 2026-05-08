find_package(gram_savitzky_golay REQUIRED)
ADD_LIBRARY(IAFEstimator "IAFEstimator.cpp")
TARGET_LINK_LIBRARIES(IAFEstimator utilities fftw3f gram_savitzky_golay::gram_savitzky_golay)
