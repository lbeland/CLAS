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

#include "filter.hpp"

using namespace dsp::filter;

IFilter::~IFilter() { unrealize(); }

std::string IFilter::description() const { return description_; }

unsigned int IFilter::nchannels() const {
  if (!realized_) {
    throw std::runtime_error("Filter has not been realized yet.");
  }
  return nchannels_;
}

bool IFilter::realized() const { return realized_; }

void IFilter::realize(unsigned int nchannels, double init) {
  if (nchannels == 0) {
    throw std::runtime_error("Number of channels cannot be 0.");
  }

  unrealize();

  if (!realize_filter(nchannels, init)) {
    throw std::runtime_error("Unable to realize filter.");
  }

  nchannels_ = nchannels;
  realized_ = true;
}

void IFilter::unrealize() {
  if (realized()) {
    unrealize_filter();
    realized_ = false;
    nchannels_ = 0;
  }
}

std::map<std::string, std::string>
dsp::filter::parse_file_header(std::istream &stream) {
  std::string expr("#\\s+([\\w\\s]*\\w)\\s*=\\s*([\\-\\w\\s,:\\.]*\\w)\\s*");
  std::regex re(expr);
  std::smatch match;

  // create minimal header
  std::map<std::string, std::string> header;
  header["filter type"] = "unknown";
  header["description"] = "";
  header["format"] = "unknown";

  bool done = false;
  std::string line;

  // file has to start with ##
  int c;
  c = stream.peek();
  if (c == EOF || c != '#') {
    throw std::runtime_error("No valid header found.");
  }

  std::getline(stream, line);
  if (line.compare(0, 2, "##") != 0) {
    throw std::runtime_error(
        "No valid header found. Missing starting sequence \"##\".");
  }

  while (!done) {
    std::getline(stream, line);

    if (line.compare(0, 1, "#") != 0) {
      throw std::runtime_error("Invalid line in header (missing \"#\" "
                               "character at beginning of line?)");
    } else if (line.compare(0, 2, "##") == 0) {
      // final line of header
      done = true;
    } else {
      if (std::regex_match(line, match, re)) {
        header[match[1].str()] = match[2].str();
      }
      // skip non matches
      // can contain arbitrary comments for documentation purposes
    }
  }

  return header;
}

IFilter *dsp::filter::construct_from_file(std::string file) {
  bool binary = false;

  std::ifstream stream(file);

  if (!stream.good()) {
    throw std::runtime_error("Cannot open filter definition file.");
  }

  auto header = parse_file_header(stream);

  if (header["format"] == "binary") {
    binary = true;

  } else if (header["format"] != "text") {
    throw std::runtime_error("Unknown filter file format.");
  }

  if (header["type"] == "fir") {
    return FirFilter::FromStream(stream, header["description"], binary);

  } else if (header["type"] == "slope") {
      return SlopeFilter::FromStream(stream, header["description"], binary);

  } else if (header["type"] == "biquad") {
    return BiquadFilter::FromStream(stream, header["description"], binary);

  } else {
    throw std::runtime_error("Unknown filter type in file.");
  }
}

IFilter *dsp::filter::construct_from_yaml(const YAML::Node &node) {
  // type: fir OR biquad OR slope
  // gain: double (biquad only)
  // coefficients: list of doubles (fir) or list of lists of 6 doubles (biquad)
  // window size : 1 uint (slope filter only)
  // order : 1 uint (slope filter only)
  // derivative order: 1 uint (slope filter only)
  // description: text

  if (node["file"]) {
    return construct_from_file(node["file"].as<std::string>());
  }

  std::string filter_type = node["type"].as<std::string>("unknown");
  std::string desc = node["description"].as<std::string>("");

  if (filter_type == "fir") {
    std::vector<double> coef = node["coefficients"].as<std::vector<double>>();
    return new FirFilter(coef, desc);

  } else if(filter_type == "slope"){
       uint32_t window_size = node["windows size"].as<unsigned int>(SlopeFilter::DEFAULT_WINDOW_SIZE);
       uint8_t derivative_order = node["derivative order"].as<unsigned int>(SlopeFilter::DEFAULT_ORDER);
       uint8_t order = node["order"].as<unsigned int>(SlopeFilter::DEFAULT_DERIVATIVE_ORDER);
       return new SlopeFilter(window_size, order, derivative_order, desc);

  } else if (filter_type == "biquad") {
    double gain = node["gain"].as<double>();
    std::vector<std::array<double, 6>> coef =
        node["coefficients"].as<std::vector<std::array<double, 6>>>();
    // std::vector<std::vector<double>> data =
    // node["coefficients"].as<std::vector<std::vector<double>>>();
    return new BiquadFilter(gain, coef, desc);

  } else {
    throw std::runtime_error("Unknown filter type in YAML.");
  }
}

FirFilter::FirFilter(const std::vector<double> &coefficients, std::string description)
    : IFilter(description), coefficients_(coefficients) {

    ntaps_ = coefficients_.size();
    if (ntaps_ < 1) {
        throw std::runtime_error("Invalid number of filter taps.");
    }
    pcoefficients_ = coefficients_.data();
}

IFilter *FirFilter::clone() {
  return new FirFilter(coefficients_, description_);
}

FirFilter *FirFilter::FromStream(std::istream &stream, std::string description,
                                 bool binary) {
  if (binary) {
    throw std::runtime_error("Binary filter files are not yet supported.");
  }

  std::vector<double> coefficients;
  double coef;

  while (stream >> coef) {
    coefficients.push_back(coef);
  }

  return new FirFilter(coefficients, description);
}

unsigned int FirFilter::order() const { return ntaps_ - 1; }

std::size_t FirFilter::group_delay() const { return order() / 2; }

double FirFilter::process_channel(double input, unsigned int channel) {
  std::rotate(registers_[channel].rbegin(), registers_[channel].rbegin() + 1,
              registers_[channel].rend());
  registers_[channel][0] = input;

  double result = 0;

  for (unsigned int k = 0; k < ntaps_; ++k) {
    result += coefficients_[k] * registers_[channel][k];
  }

  return result;
}

void FirFilter::process_sample(std::vector<double> &input,
                               std::vector<double> &output) {
  double result;

  for (unsigned int channel = 0; channel < nchannels_; ++channel) {
    std::rotate(registers_[channel].rbegin(), registers_[channel].rbegin() + 1,
                registers_[channel].rend());
    registers_[channel][0] = input[0];

    result = 0;

    for (unsigned int k = 0; k < ntaps_; ++k) {
      result += coefficients_[k] * registers_[channel][k];
    }

    output[channel] = result;
  }
}

void FirFilter::process_sample(std::vector<double>::iterator input,
                               std::vector<double>::iterator output) {
  double result;

  for (unsigned int channel = 0; channel < nchannels_; ++channel) {
    std::rotate(registers_[channel].rbegin(), registers_[channel].rbegin() + 1,
                registers_[channel].rend());
    registers_[channel][0] = *input++;

    result = 0;

    for (unsigned int k = 0; k < ntaps_; ++k) {
      result += coefficients_[k] * registers_[channel][k];
    }

    *output = result;
    output++;
  }
}

void FirFilter::process_sample(double *input, double *output) {
  // skip check if filter has been realized??
  double result;
  double *coef, *reg;

  for (unsigned int channel = 0; channel < nchannels_; ++channel) {
    reg = pregisters_[channel];

    memmove((void *)(reg + 1), (void *)reg, (ntaps_ - 1) * sizeof(double));
    *reg = *input;

    coef = pcoefficients_;
    result = 0;

    for (unsigned int k = 0; k < ntaps_; ++k) {
      result += *coef++ * *reg++;
    }

    *output = result;

    input++;   // move to next channel
    output++;  // move to next channel
  }
}

void FirFilter::process_channel(std::vector<double> &input,
                                std::vector<double> &output,
                                unsigned int channel) {
  double result;

  uint64_t nsamples = input.size();
  // check output.size() == input.size()

  for (uint64_t s = 0; s < nsamples; ++s) {
    std::rotate(registers_[channel].rbegin(), registers_[channel].rbegin() + 1,
                registers_[channel].rend());
    registers_[channel][0] = input[s];

    result = 0;

    for (unsigned int k = 0; k < ntaps_; ++k) {
      result += coefficients_[k] * registers_[channel][k];
    }

    output[s] = result;
  }
}

void FirFilter::process_channel(uint64_t nsamples,
                                std::vector<double>::iterator input,
                                std::vector<double>::iterator output,
                                unsigned int channel) {
  double result;

  for (uint64_t s = 0; s < nsamples; ++s) {
    std::rotate(registers_[channel].rbegin(), registers_[channel].rbegin() + 1,
                registers_[channel].rend());
    registers_[channel][0] = *input++;

    result = 0;

    for (unsigned int k = 0; k < ntaps_; ++k) {
      result += coefficients_[k] * registers_[channel][k];
    }

    *output = result;
    output++;
  }
}

void FirFilter::process_channel(uint64_t nsamples, double *input,
                                double *output, unsigned int channel) {
  throw std::runtime_error("Not yet implemented.");
}

void FirFilter::process_by_channel(std::vector<std::vector<double>> &input,
                                   std::vector<std::vector<double>> &output) {
  uint64_t nsamples = input.size();
  for (uint64_t s = 0; s < nsamples; ++s) {
    process_sample(input[s], output[s]);
  }
}

void FirFilter::process_by_sample(std::vector<std::vector<double>> &input,
                                  std::vector<std::vector<double>> &output) {
  for (unsigned int c = 0; c < nchannels_; ++c) {
    process_channel(input[c], output[c]);
  }
}

void FirFilter::process_by_channel(uint64_t nsamples, double **input,
                                   double **output) {
  for (uint64_t s = 0; s < nsamples; ++s) {
    process_sample(input[s], output[s]);
  }
}

void FirFilter::process_by_sample(uint64_t nsamples, double **input,
                                  double **output) {
  for (unsigned int c = 0; c < nchannels_; ++c) {
    process_channel(nsamples, input[c], output[c]);
  }
}

void FirFilter::process_by_channel(uint64_t nsamples,
                                   std::vector<double> &input,
                                   std::vector<double> &output) {
  assert(nsamples * nchannels_ == input.size() &&
         input.size() == output.size());
  auto in_it = input.begin();
  auto out_it = output.begin();
  for (unsigned int s = 0; s < nsamples; ++s) {
    process_sample(in_it, out_it);
    in_it += nchannels_;
    out_it += nchannels_;
  }
}

void FirFilter::process_by_sample(uint64_t nsamples, std::vector<double> &input,
                                  std::vector<double> &output) {
  assert(nsamples * nchannels_ == input.size() &&
         input.size() == output.size());
  auto in_it = input.begin();
  auto out_it = output.begin();
  for (unsigned int c = 0; c < nchannels_; ++c) {
    process_channel(nsamples, in_it, out_it, c);
    in_it += nsamples;
    out_it += nsamples;
  }
}

bool FirFilter::realize_filter(unsigned int nchannels, double init) {
  // create and initialize register for each channel
  for (unsigned int k = 0; k < nchannels; ++k) {
    registers_.push_back(std::vector<double>(ntaps_, init));
    pregisters_.push_back(registers_.back().data());
  }

  return true;
}

void FirFilter::unrealize_filter() {
  pregisters_.clear();
  registers_.clear();
}


SlopeFilter *SlopeFilter::FromStream(std::istream &stream,
                               std::string description,
                               bool binary){
    if (binary) {
        throw std::runtime_error("Binary filter files are not yet supported.");
    }

    uint32_t window_size, order, derivative_order;

    stream >> window_size;
    stream >> order;
    stream >> derivative_order;

    return new SlopeFilter(window_size, order, derivative_order, description);
};

BiquadFilter::BiquadFilter(double gain,
                           std::vector<std::array<double, 6>> &coefficients,
                           std::string description)
    : IFilter(description), gain_(gain), coefficients_(coefficients) {

  nstages_ = coefficients_.size();
  if (nstages_ < 1) {
    throw std::runtime_error("Invalid number of biquad stages.");
  }
}

IFilter *BiquadFilter::clone() {
  return new BiquadFilter(gain_, coefficients_, description_);
}

BiquadFilter *BiquadFilter::FromStream(std::istream &stream,
                                       std::string description,
                                       bool binary = false) {
  if (binary) {
    throw std::runtime_error("Binary filter files are not yet supported.");
  }

  double gain;
  if (!(stream >> gain)) {
    throw std::runtime_error("No filter coefficients found.");
  }

  // read all data
  std::vector<double> data;
  double value;

  while (stream >> value) {
    data.push_back(value);
  }

  // determine number of biquad stages
  if (data.size() % 6 != 0) {
    throw std::runtime_error(
        "Each biquad stage requires exactly 6 coefficients");
  }

  unsigned int nstages = data.size() / 6;
  std::vector<std::array<double, 6>> coefficients(nstages);

  auto data_it = data.begin();

  for (unsigned int stage = 0; stage < nstages; ++stage) {
    for (unsigned int coef = 0; coef < 6; ++coef) {
      coefficients[stage][coef] = (*data_it++);
    }
  }

  return new BiquadFilter(gain, coefficients, description);
}

unsigned int BiquadFilter::order() const { return nstages_ * 2; }

double BiquadFilter::process_channel(double x, unsigned int c) {
  double u_n, y_n = 0.0;

  for (unsigned int s = 0; s < nstages_; ++s) {
    u_n = x - coefficients_[s][4] * registers_[c][s][0] -
          coefficients_[s][5] * registers_[c][s][1];
    y_n = coefficients_[s][0] * u_n +
          coefficients_[s][1] * registers_[c][s][0] +
          coefficients_[s][2] * registers_[c][s][1];
    registers_[c][s][1] = registers_[c][s][0];
    registers_[c][s][0] = u_n;
    x = y_n;
  }

  return y_n * gain_;
}

void BiquadFilter::process_sample(std::vector<double>::iterator input,
                                  std::vector<double>::iterator output) {
  if (!realized_) {
    throw std::runtime_error("Filter has not been realized yet.");
  }

  for (unsigned int c = 0; c < nchannels_; ++c) {
    *output = process_channel((*input++), c);
    output++;
  }
}

void BiquadFilter::process_sample(double *input, double *output) {
  if (!realized_) {
    throw std::runtime_error("Filter has not been realized yet.");
  }

  for (unsigned int c = 0; c < nchannels_; ++c) {
    *output = process_channel((*input++), c);
    output++;
  }
}

void BiquadFilter::process_channel(std::vector<double> &input,
                                   std::vector<double> &output,
                                   unsigned int channel) {
  uint64_t nsamples = input.size();

  for (uint64_t s = 0; s < nsamples; ++s) {
    output[s] = process_channel(input[s], channel);
  }
}

void BiquadFilter::process_channel(uint64_t nsamples,
                                   std::vector<double>::iterator input,
                                   std::vector<double>::iterator output,
                                   unsigned int channel) {
  for (uint64_t s = 0; s < nsamples; ++s) {
    *output = process_channel((*input++), channel);
    output++;
  }
}

void BiquadFilter::process_channel(uint64_t nsamples, double *input,
                                   double *output, unsigned int channel) {
  for (uint64_t s = 0; s < nsamples; ++s) {
    output[s] = process_channel(input[s], channel);
  }
}

void BiquadFilter::process_by_channel(
    std::vector<std::vector<double>> &input,
    std::vector<std::vector<double>> &output) {

  uint64_t nsamples = input.size();
  for (uint64_t s = 0; s < nsamples; ++s) {
    process_sample(input[s], output[s]);
  }
}

void BiquadFilter::process_by_sample(std::vector<std::vector<double>> &input,
                                     std::vector<std::vector<double>> &output) {
  for (unsigned int c = 0; c < nchannels_; ++c) {
    process_channel(input[c], output[c]);
  }
}

void BiquadFilter::process_by_channel(uint64_t nsamples, double **input,
                                      double **output) {
  for (uint64_t s = 0; s < nsamples; ++s) {
    process_sample(input[s], output[s]);
  }
}

void BiquadFilter::process_by_sample(uint64_t nsamples, double **input,
                                     double **output) {
  for (unsigned int c = 0; c < nchannels_; ++c) {
    process_channel(nsamples, input[c], output[c], c);
  }
}

void BiquadFilter::process_by_channel(uint64_t nsamples,
                                      std::vector<double> &input,
                                      std::vector<double> &output) {
  assert(nsamples * nchannels_ == input.size() &&
         input.size() == output.size());
  auto in_it = input.begin();
  auto out_it = output.begin();
  for (unsigned int s = 0; s < nsamples; ++s) {
    process_sample(in_it, out_it);
    in_it += nchannels_;
    out_it += nchannels_;
  }
}

void BiquadFilter::process_by_sample(uint64_t nsamples,
                                     std::vector<double> &input,
                                     std::vector<double> &output) {
  assert(nsamples * nchannels_ == input.size() &&
         input.size() == output.size());
  auto in_it = input.begin();
  auto out_it = output.begin();
  for (unsigned int c = 0; c < nchannels_; ++c) {
    process_channel(nsamples, in_it, out_it, c);
    in_it += nsamples;
    out_it += nsamples;
  }
}

bool BiquadFilter::realize_filter(unsigned int nchannels, double init) {
  for (unsigned int k = 0; k < nchannels; ++k) {
    registers_.push_back(std::vector<std::array<double, 2>>(nstages_));
    for (auto &it : registers_.back()) {
      it.fill(init);
    }
  }

  return true;
}

void BiquadFilter::unrealize_filter() { registers_.clear(); }
