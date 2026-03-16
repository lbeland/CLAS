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

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {

  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T, class A> T join(const A &begin, const A &end, const T &t) {

  T result;
  A it = begin;

  if (it != end) {
    result.append(*it++);
  }

  for (; it != end; ++it) {
    result.append(t).append(*it);
  }

  return result;
}
