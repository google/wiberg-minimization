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

#include "l1/linear_system/basis.h"
#include "coin/ClpSimplex.hpp"
#include "eigen3/Eigen/SparseCore"
#include "l1/linear_system/linear_program.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace wiberg {

void Basis::Compute(const LinearProgram &linear_program) {
  if (linear_program.algorithm() != ClpSolve::useDual) {
    std::cerr << "Can only compute basis for dual solutions." << std::endl;
    abort();
  }

  // Get the basis row and column indices.
  const MatrixXd &A = linear_program.A();
  const ClpSimplex &solver = linear_program.solver();
  basic_column_indices_.clear();
  for (int i = 0; i < A.cols(); ++i) {
    // TODO(strelow): Reverify that ClpSimplex::basic is the only type that
    // needs to be added to the basis here.
    if (solver.getColumnStatus(i) == ClpSimplex::basic) {
      basic_column_indices_.push_back(i);
    }
  }
  basic_row_indices_.clear();
  for (int i = 0; i < A.rows(); ++i) {
    if (solver.getRowStatus(i) == ClpSimplex::basic) {
      basic_row_indices_.push_back(i);
    }
  }

  // Make an explicit mapping from A's columns to the basis's columns.  -1
  // means that the A column doesn't have a corresponding column in the
  // basis.
  A_column_to_basis_column_map_.assign(A.cols(), -1);
  for (int i = 0; i < basic_column_indices_.size(); ++i) {
    const int basic_column_index = basic_column_indices_[i];
    assert(basic_column_index >= 0);
    assert(basic_column_index < A_column_to_basis_column_map_.size());
    A_column_to_basis_column_map_[basic_column_index] = i;
  }

  // Use the indices to get the basis and the basic solution.
  const VectorXd &x = linear_program.x();
  x_B_ = VectorXd::Zero(A.rows());
  B_ = MatrixXd::Zero(A.rows(), A.rows());
  int count = 0;
  for (int i = 0; i < basic_column_indices_.size(); ++i) {
    x_B_(count) = x(basic_column_indices_[i]);
    B_.col(count) = A.col(basic_column_indices_[i]);
    ++count;
  }
  for (int i = 0; i < basic_row_indices_.size(); ++i) {
    x_B_(count) = x(A.cols() + basic_row_indices_[i]);
    B_(basic_row_indices_[i], count) = 1.0;
    ++count;
  }
  assert(count == A.rows());

  // Compute the inverse basis.
  B_inverse_ = B_.inverse();
}

}  // namespace wiberg
