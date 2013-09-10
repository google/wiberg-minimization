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

// Routines for minimizing the L1 residual of an overdetermined linear system,
// Cy = d.  The approach is described in Section 3.2 (Linear L1 Minimization)
// and (for trust regions) Section 3.3 equation (14) of the CVPR 2012 paper.
// The paper is here: http://research.google.com/pubs/pub37749.html.

#ifndef WIBERG_L1_LINEAR_SYSTEM_LINEAR_SYSTEM_L1_H_
#define WIBERG_L1_LINEAR_SYSTEM_LINEAR_SYSTEM_L1_H_

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SparseCore"
#include "coin/ClpSimplex.hpp"

namespace wiberg {

class LinearSystemL1 {
 public:
  // Solve() minimizes the L1 residual of an overdetermined linear system,
  // Cy = d.  It recasts the problem as a linear program using
  // GetLinearProgram() below.
  //
  // mu >= 0 is the trust region size.  If mu > 0, the solution is constrained
  // to have L1 norm less than or equal to mu.  If mu = 0, then no trust region
  // constraint is used.  The trust region is needed to ensure convergence
  // in successive linear programming, where it is automatically adjusted to
  // find a step that decreases the objective.
  //
  // algorithm specifies the algorithm to use in solving the linear program,
  // either ClpSolve::usePrimal or ClpSolve::useDual.  Dual is usually faster,
  // so use dual unless you have a hard time limit, as described next.
  //
  // time_limit != -1 specifies a solve time limit in milliseconds, and
  // time_limit = -1 meaning no time limit.  A time limit can be used with the
  // primal algorithm but not the dual algorithm.
  static void Solve(
      const Eigen::SparseMatrix<double> &C,
      const Eigen::VectorXd &d,
      double mu,
      ClpSolve::SolveType algorithm,
      double time_limit,
      Eigen::VectorXd *y);
  // Given the linear system Cy = d, constructs the linear program components
  // A, b, c whose solution x can be converted into a y that minimizes the L1
  // residual of the linear system.
  static void GetLinearProgram(
      const Eigen::SparseMatrix<double> &C,
      const Eigen::VectorXd &d,
      double mu,
      Eigen::SparseMatrix<double> *A,
      Eigen::VectorXd *b,
      Eigen::VectorXd *c);
  // Extract the solution to Cy = d from the linear program solution x.
  static void GetYFromX(const Eigen::VectorXd &x, Eigen::VectorXd *y);
};

}  // namespace wiberg

#endif  // WIBERG_L1_LINEAR_SYSTEM_LINEAR_SYSTEM_L1_H_
