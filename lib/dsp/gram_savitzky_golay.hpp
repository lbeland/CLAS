// ---------------------------------------------------------------------
// BSD 2-Clause License
//
// Copyright (c) 2012-2019, CNRS-UM LIRMM, CNRS-AIST JRL
// All rights reserved.
// reference: https://github.com/arntanguy/gram_savitzky_golay
//
// Modified in 2021 by Neuro-Electronics Research Flanders
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------

#pragma once

#include <vector>
#include <cassert>
#include <cmath>

namespace gram_sg
{

inline double compute_gram_polynomial(const int i, const int m, const int k, const int s){
    if(k>0)
    {
        return (4.*k-2.) / (k*(2.*m - k+1.)) * (i * compute_gram_polynomial(i, m, k - 1, s) + s * compute_gram_polynomial(i, m, k-1, s-1))
                - ((k-1.) * (2.*m + k)) / (k * (2.*m - k+1.)) * compute_gram_polynomial(i, m, k-2, s);
    }

    if(k == 0 && s == 0){
        return 1.;
    }else{
        return 0.;
    }

};


inline double compute_generalized_factorial(const int a, const int b){
    double gf = 1.;

    for(int j=(a-b)+1; j<=a; j++)
    {
        gf*=j;
    }
    return gf;
};


inline double compute_weight(const int i, const int t, const int m, const int n, const int s){
    double w = 0;
    for(int k = 0; k <= n; ++k)
    {
        w = w + (2*k+1) * (compute_generalized_factorial(2*m, k)/compute_generalized_factorial(2*m+k+1, k+1))
                * compute_gram_polynomial(i, m, k, 0)* compute_gram_polynomial(t, m, k, s);
    }
    return w;
};


inline std::vector<double> compute_weights(const int m, const int t, const int n, const int s){
    std::vector<double> weights(2*static_cast<size_t>(m)+1);

    for(int i=0; i<2*m+1; ++i)
    {
        weights[static_cast<size_t>(i)] = compute_weight(i - m, t, m, n, s);
    }
    return weights;
};
} // namespace gram_sg
