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

// LinearProgram is a wrapper that uses CLP to solve a linear program, and
// retains the linear program and solver so they can be used in computing the
// derivatives needed for Wiberg minimization.

#ifndef WIBERG_L1_LINEAR_SYSTEM_LINEAR_PROGRAM_H_
#define WIBERG_L1_LINEAR_SYSTEM_LINEAR_PROGRAM_H_

#include <memory>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SparseCore"
#include "coin/ClpSimplex.hpp"

class ClpSimplex;

namespace wiberg {

class LinearProgram {
 public:
  LinearProgram();
  ~LinearProgram();
  // A, b, and c are the linear program.
  const Eigen::SparseMatrix<double> &A() const { return A_; }
  const Eigen::VectorXd &b() const { return b_; }
  const Eigen::VectorXd &c() const { return c_; }
  ClpSolve::SolveType algorithm() const { return algorithm_; }
  // x is the solution.
  const ClpSimplex &solver() const { return *solver_; }
  ClpSimplex *solver() { return solver_.get(); }
  const Eigen::VectorXd &x() const { return x_; }
  // Solve the linear program specified by A, b, and c.  Choose -1.0 as the
  // time limit to specify no time limit.
  void Solve(const Eigen::SparseMatrix<double> &A,
             const Eigen::VectorXd &b,
             const Eigen::VectorXd &c,
             ClpSolve::SolveType algorithm,
             double time_limit);

 private:
  Eigen::SparseMatrix<double> A_;
  Eigen::VectorXd b_, c_, x_;
  std::unique_ptr<ClpSimplex> solver_;
  ClpSolve::SolveType algorithm_;
};

}  // namespace wiberg

#endif  // WIBERG_L1_LINEAR_SYSTEM_LINEAR_PROGRAM_H_
