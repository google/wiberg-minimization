// Copyright 2013 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "l1/factorization/solve_V.h"
#include "l1/factorization/solve_V_j.h"

using Eigen::MatrixXd;
using std::unique_ptr;

namespace wiberg {

SolveV::SolveV() {}

SolveV::~SolveV() {
}

const SolveVj &SolveV::solve_V_j(int i) const {
  assert(i >= 0);
  assert(i < size_);
  return solve_V_j_[i];
}

void SolveV::CopyFrom(const SolveV &solve_V) {
  size_ = solve_V.size();
  solve_V_j_.reset(new SolveVj[size_]);
  for (int i = 0; i < size_; ++i) {
    solve_V_j_[i].CopyFrom(solve_V.solve_V_j(i));
  }
  V_ = solve_V.V_;
}

void SolveV::Solve(const MatrixXd &U, const MatrixXd &Y,
                   const MatrixXd &W_hat) {
  size_ = Y.cols();
  solve_V_j_.reset(new SolveVj[size_]);
  V_ = MatrixXd::Zero(U.cols(), size_);
  for (int j = 0; j < size_; ++j) {
    solve_V_j_[j].Solve(U, Y, W_hat, j);
    V_.col(j) = solve_V_j_[j].V_j();
  }
}

}  // namespace wiberg
