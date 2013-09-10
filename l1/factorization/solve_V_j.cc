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

#include "l1/factorization/solve_V_j.h"
#include "l1/linear_system/linear_system_l1_derivatives.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace wiberg {

void SolveVj::CopyFrom(const SolveVj &solve_V_j) {
  observed_ = solve_V_j.observed_;
  V_j_ = solve_V_j.V_j_;
  derivative_V_j_wrt_U_ = solve_V_j.derivative_V_j_wrt_U_;
}

void SolveVj::Solve(const MatrixXd &U, const MatrixXd &Y,
                    const MatrixXd &observation_mask, int j) {
  // Find the observations that are present in Y's column j.
  for (int i = 0; i < observation_mask.rows(); ++i) {
    if (observation_mask(i, j) == 1.0) {
      observed_.push_back(i);
    }
  }

  // Store those entries of Y in y_pruned and the corresponding rows of U
  // in U_pruned.
  VectorXd y_pruned(observed_.size());
  MatrixXd U_pruned(observed_.size(), U.cols());
  for (int i = 0; i < observed_.size(); ++i) {
    y_pruned(i) = Y(observed_[i], j);
    U_pruned.row(i) = U.row(observed_[i]);
  }

  // Solve for V_j and get dV_j/dU_pruned.
  LinearSystemL1Derivatives linear_system_l1_derivatives;
  const double mu = 0;  // No trust region constraint.
  linear_system_l1_derivatives.Solve(U_pruned, y_pruned, mu);
  V_j_ = linear_system_l1_derivatives.y();
  const MatrixXd &derivative_V_j_wrt_U_pruned =
      linear_system_l1_derivatives.derivative_y_wrt_C();

  // Zero-pad dV_j/dU_pruned to get dV_j/dU.
  derivative_V_j_wrt_U_ = MatrixXd::Zero(V_j_.rows(), U.rows() * U.cols());
  for (int i = 0; i < observed_.size(); ++i) {
    for (int j = 0; j < U.cols(); ++j) {
      const int old_column = j * observed_.size() + i;
      const int new_column = j * U.rows() + observed_[i];
      derivative_V_j_wrt_U_.col(new_column) =
          derivative_V_j_wrt_U_pruned.col(old_column);
    }
  }
}

}  // namespace wiberg
