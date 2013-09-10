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

// Extracts and stores information about the basis from a linear program
// solution.  These are used in computing the derivatives of the linear
// program solution x with respect to the coefficient matrix A and right-hand
// side b.
//
// As described in linear_program.h, Wiberg minimization solves two different
// kinds of linear programs, "inner" and "outer."  The basis information here
// only needs to be extracted for the "inner" linear programs, so we've
// separated it out into this class instead of extracting the basis in
// the LinearProgram class.

#ifndef WIBERG_L1_LINEAR_SYSTEM_BASIS_H_
#define WIBERG_L1_LINEAR_SYSTEM_BASIS_H_

#include <vector>
#include "eigen3/Eigen/Dense"

namespace wiberg {

class LinearProgram;

class Basis {
 public:
  Basis() {}
  ~Basis() {}
  // B is the basis.
  const Eigen::MatrixXd &B() const { return B_; }
  // B_inverse is the inverse basis.
  const Eigen::MatrixXd &B_inverse() const { return B_inverse_; }
  // x_B is the basic solution.
  const Eigen::VectorXd &x_B() const { return x_B_; }
  // basic_row_indices gives the rows of A of included in the basis B.
  const std::vector<int> &basic_row_indices() const { return basic_row_indices_; }
  // basic_column_indices gives the columns of A included in the basis B.
  const std::vector<int> &basic_column_indices() const {
    return basic_column_indices_;
  }
  // Maps columns in A to columns in the basis B.  If column i of A isn't
  // included in the basis, A_column_to_basis_column_map_[i] is -1.
  const std::vector<int> &A_column_to_basis_column_map() const {
    return A_column_to_basis_column_map_;
  }
  void Compute(const LinearProgram &linear_program);

 private:
  Eigen::MatrixXd B_, B_inverse_;
  Eigen::VectorXd x_B_;
  std::vector<int> basic_row_indices_,
    basic_column_indices_,
    A_column_to_basis_column_map_;
};

}  // namespace wiberg

#endif  // WIBERG_L1_LINEAR_SYSTEM_BASIS_H_
