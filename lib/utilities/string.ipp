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

template <typename T>
std::string join(const std::vector<T> &s, std::string delim) {
  std::stringstream ss;

  for (auto it = s.begin(); it != s.end(); ++it) {
    ss << *it;
    if (std::next(it) != s.end()) {
      ss << delim;
    }
  }

  return ss.str();
}

template <typename T> std::string to_string_n(const T a_value, const int n) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(n) << a_value;
  return out.str();
}

template <typename T> T from_string(std::string s) {
  std::stringstream ss(s);
  T result;
  ss >> result;
  return result;
}
