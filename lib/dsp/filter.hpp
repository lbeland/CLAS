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

#include "gram_savitzky_golay.hpp"

#include <yaml-cpp/yaml.h>

#include <array>
#include <vector>
#include <cassert>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <algorithm>
#include <cctype>
#include <exception>


namespace dsp {
namespace filter {

class IFilter {
public:
    IFilter(std::string description) : description_(description) {}

    virtual ~IFilter();

    std::string description() const;

    virtual IFilter *clone() = 0;
    virtual unsigned int order() const = 0;

    unsigned int nchannels() const;
    bool realized() const;
    void realize(unsigned int nchannels, double init = 0.0);
    void unrealize();

    // single channel, single sample
    virtual double process_channel(double, unsigned int channel) = 0;

    // all channels, single sample
    virtual void process_sample(std::vector<double> &, std::vector<double> &) = 0;
    virtual void process_sample(std::vector<double>::iterator,
                                std::vector<double>::iterator) = 0;
    virtual void process_sample(double *, double *) = 0;

    // single channel, multiple samples
    virtual void process_channel(std::vector<double> &, std::vector<double> &,
                                 unsigned int channel = 0) = 0;
    virtual void process_channel(uint64_t nsamples, std::vector<double>::iterator,
                                 std::vector<double>::iterator,
                                 unsigned int channel = 0) = 0;
    virtual void process_channel(uint64_t nsamples, double *, double *,
                                 unsigned int channel) = 0;

    // all channels, multiple samples
    virtual void process_by_channel(
            std::vector<std::vector<double>> &,
            std::vector<std::vector<double>> &) = 0;  // samples<channels>
    virtual void process_by_sample(
            std::vector<std::vector<double>> &,
            std::vector<std::vector<double>> &) = 0;  // channels<samples>
    virtual void process_by_channel(uint64_t nsamples, double **,
                                    double **) = 0;  // samples<channels>
    virtual void process_by_sample(uint64_t nsamples, double **,
                                   double **) = 0;  // channels<samples>
    virtual void
    process_by_channel(uint64_t nsamples, std::vector<double> &,
                       std::vector<double> &) = 0;  // samples<channels>
    virtual void
    process_by_sample(uint64_t nsamples, std::vector<double> &,
                      std::vector<double> &) = 0;  // channels<samples>

protected:
    virtual bool realize_filter(unsigned int nchannels, double init) = 0;
    virtual void unrealize_filter() {};

protected:
    std::string description_;
    bool realized_ = false;
    unsigned int nchannels_ = 0;
};

class FirFilter : public IFilter {
public:
    FirFilter(const std::vector<double> &coefficients, std::string description = "");
    virtual IFilter *clone();

    static FirFilter *FromStream(std::istream &stream, std::string description,
                                 bool binary = false);

    unsigned int order() const final;

    std::size_t group_delay() const;

    // single channel, single sample
    double process_channel(double input, unsigned int channel = 0) final;

    // all channels, single sample
    void process_sample(std::vector<double> &input,
                        std::vector<double> &output) final;
    void process_sample(std::vector<double>::iterator input,
                        std::vector<double>::iterator output) final;
    void process_sample(double *input, double *output) final;

    // single channel, multiple samples
    void process_channel(std::vector<double> &input, std::vector<double> &output,
                         unsigned int channel = 0) final;
    void process_channel(uint64_t nsamples, std::vector<double>::iterator input,
                         std::vector<double>::iterator output,
                         unsigned int channel = 0);
    void process_channel(uint64_t nsamples, double *input, double *output,
                         unsigned int channel = 0);

    // all channels, multiple samples
    void
    process_by_channel(std::vector<std::vector<double>> &input,
                       std::vector<std::vector<double>> &output) final;
    void
    process_by_sample(std::vector<std::vector<double>> &input,
                      std::vector<std::vector<double>> &output) final;
    void process_by_channel(uint64_t nsamples, double **input,
                            double **output) final;
    void process_by_sample(uint64_t nsamples, double **input,
                           double **output) final;

    virtual void process_by_channel(uint64_t nsamples, std::vector<double> &input,
                                    std::vector<double> &output);
    virtual void process_by_sample(uint64_t nsamples, std::vector<double> &input,
                                   std::vector<double> &output);

protected:
    bool realize_filter(unsigned int nchannels,
                        double init = 0.0) final;
    void unrealize_filter() final;

protected:
    std::vector<double> coefficients_;
    double *pcoefficients_;
    unsigned int ntaps_;

    std::vector<std::vector<double>> registers_;
    std::vector<double *> pregisters_;

    double *preg_;
};

class SlopeFilter : public FirFilter {
public:

    SlopeFilter(uint32_t window_size, uint32_t order, uint32_t derivative_order, std::string description=""):
        FirFilter(gram_sg::compute_weights(window_size, window_size, order, derivative_order), description),
        window_size_(window_size), order_(order), derivative_order_(derivative_order){};

    static SlopeFilter *FromStream(std::istream &stream,
                                   std::string description,
                                   bool binary);

    virtual IFilter *clone(){
        return new SlopeFilter(window_size_, order_, derivative_order_);
    };

protected:
    size_t window_size_;
    uint32_t order_;
    uint32_t derivative_order_;
public:
    static constexpr uint16_t DEFAULT_WINDOW_SIZE = 4;
    static constexpr uint8_t DEFAULT_ORDER = 1;
    static constexpr uint8_t DEFAULT_DERIVATIVE_ORDER = 1;
};

class BiquadFilter : public IFilter {
public:
    BiquadFilter(double gain, std::vector<std::array<double, 6>> &coefficients,
                 std::string description = "");
    virtual IFilter *clone();

    static BiquadFilter *FromStream(std::istream &stream, std::string description,
                                    bool binary);

    unsigned int order() const final;

    // single channel, single sample
    double process_channel(double x, unsigned int c = 0) final;

    // all channels, single sample
    void process_sample(std::vector<double> &input,
                        std::vector<double> &output) final{};
    void process_sample(std::vector<double>::iterator input,
                        std::vector<double>::iterator output) final;
    void process_sample(double *input, double *output) final;

    // single channel, multiple samples
    void process_channel(std::vector<double> &input, std::vector<double> &output,
                         unsigned int channel = 0) final;
    void process_channel(uint64_t nsamples, std::vector<double>::iterator input,
                         std::vector<double>::iterator output,
                         unsigned int channel = 0) final;
    void process_channel(uint64_t nsamples, double *input, double *output,
                         unsigned int channel = 0) final;

    // all channels, multiple samples
    void
    process_by_channel(std::vector<std::vector<double>> &input,
                       std::vector<std::vector<double>> &output) final;
    void
    process_by_sample(std::vector<std::vector<double>> &input,
                      std::vector<std::vector<double>> &output) final;
    void process_by_channel(uint64_t nsamples, double **input,
                            double **output) final;
    void process_by_sample(uint64_t nsamples, double **input,
                           double **output) final;

    virtual void process_by_channel(uint64_t nsamples, std::vector<double> &input,
                                    std::vector<double> &output);
    virtual void process_by_sample(uint64_t nsamples, std::vector<double> &input,
                                   std::vector<double> &output);

protected:
    bool realize_filter(unsigned int nchannels,
                        double init = 0.0) final;
    void unrealize_filter() final;

protected:
    double gain_;
    std::vector<std::array<double, 6>> coefficients_;

    unsigned int nstages_;

    std::vector<std::vector<std::array<double, 2>>> registers_;
};

std::map<std::string, std::string> parse_file_header(std::istream &stream);
IFilter *construct_from_file(std::string file);

IFilter *construct_from_yaml(const YAML::Node &node);

}  // namespace filter
}  // namespace dsp
