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

// Routines for computing the derivatives dy/dC and dy/dd of the solution of
// the overdetermined L1 system Cy = d.
//
// The linear system is described in section 3.2 ("Linear L1 Minimization")
// of the Strelow CVPR 2012 paper, (http://research.google.com/pubs/
// pub37749.html).  That section skips the details on computing the derivatives,
// saying "...we can then get dy/dC and dy/dd...with some simple algebra and
// rearranging."  Here is a more detailed description, with pointers to
// the routines below that compute each intermediate result.
//
// As described in section 3.2, the linear system Cy = d is recast as a linear
// program with components A, b, c and a solution x that encodes y.  dy/dC is
// then computed as follows:
//
//   DerivativeYWrtC computes dy/dC.  x = [y+ y- t] and y = y+ - y-, so
//   it uses dx/dC from:
//
//   DerivativeXWrtC computes dx/dC.  A = [C -C I; -C C I], so it uses
//   dx/dA from:
//
//   DerivativeXWrtA computes dx/dA.  dx/dA is a zero padded version of
//   dx_B/dB, from:
//
//   DerivativeXBWrtB computes dx_B/dB, given by equation (3) of the paper.
//
// The analogous chain for dy/dd is:
//
//   DerivativeYWrtD computes dy/dd.  x = [y+ y- t] and y = y+ - y-, so
//   it uses dx/dd from:
//
//   DerivativeXWrtD computes dx/dd.  b = [d -d], so it uses dx/db from:
//
//   DerivativeXWrtb computes dx/db.  dx/db is a zero padded version of
//   dx_B/db, from:
//
//   DerivativeXBWrtb computes dx_B/db, given by equation (4) of the paper.
//
// While DerivativeYWrtC() and DerivativeYWrtD() compute the full matrices
// dy/dC and dy/dd, the other routines only compute individual components
// of the other derivatives.  These other matrices can be extremely large and
// most of their elements aren't needed to find dy/dC and dy/dd.
//
// We represent derivatives with respect to matrices by flattening the matrices
// by column.  So for example, if C has num_rows rows, then the first num_rows
// columns of dy/dC correspond to the num_rows components in the first column
// of C.

#ifndef WIBERG_L1_LINEAR_SYSTEM_LINEAR_SYSTEM_L1_DERIVATIVES_H_
#define WIBERG_L1_LINEAR_SYSTEM_LINEAR_SYSTEM_L1_DERIVATIVES_H_

#include "eigen3/Eigen/Dense"
#include "l1/linear_system/basis.h"
#include "l1/linear_system/linear_program.h"

namespace wiberg {

class Basis;
class LinearProgram;

class LinearSystemL1Derivatives {
 public:
  LinearSystemL1Derivatives() {}
  ~LinearSystemL1Derivatives() {}
  const Eigen::VectorXd &y() const { return y_; }
  const Eigen::MatrixXd &derivative_y_wrt_C() const
    { return derivative_y_wrt_C_; }
  const Eigen::MatrixXd &derivative_y_wrt_d() const
    { return derivative_y_wrt_d_; }

  // Similar to Solve() in LinearSystemL1, but also computes dy/dC and dy/dd.
  // This version of Solve() would be used only for Wiberg's "inner" solve,
  // so we assume the linear program is solved using the dual algorithm without
  // a timeout, so those parameters are omitted here.
  double Solve(const Eigen::MatrixXd &C, const Eigen::VectorXd &d, double mu);

 private:
  // These functions compute dy/dC, as described in the comment at the top of
  // the file.
  void DerivativeYWrtC(const Eigen::MatrixXd &C);
  double DerivativeXWrtC(const Eigen::MatrixXd &C,
                         int x_component, int C_row, int C_column);
  double DerivativeXWrtA(int x_component, int A_row, int A_column);
  double DerivativeXBWrtB(int x_B_component, int B_row, int B_column);

  // These functions compute dy/dd, as described in the comment at the top of
  // the file.  In the function names here, wrtb means "with respect to (small)
  // b", where b is the right-hand side vector in the linear program, in
  // contrast to the basis (big) B.
  void DerivativeYWrtD(const Eigen::VectorXd &d);
  double DerivativeXWrtD(int x_component, int d_component, int d_rows);
  double DerivativeXWrtb(int x_component, int b_component);
  double DerivativeXBWrtb(int x_B_component, int b_component);

  LinearProgram linear_program_;
  Basis basis_;
  Eigen::VectorXd y_;
  Eigen::MatrixXd derivative_y_wrt_C_, derivative_y_wrt_d_;
};

}  // namespace wiberg

#endif  // WIBERG_L1_LINEAR_SYSTEM_LINEAR_SYSTEM_L1_DERIVATIVES_H_
