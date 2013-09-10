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

#include "l1/linear_system/linear_system_l1_derivatives.h"
#include <vector>
#include "l1/linear_system/linear_program.h"
#include "l1/linear_system/linear_system_l1.h"

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;

namespace wiberg {

double LinearSystemL1Derivatives::Solve(const MatrixXd &C,
                                        const VectorXd &d,
                                        double mu) {
  // Convert the problem to a linear program.
  SparseMatrix<double> A;
  VectorXd b, c;
  LinearSystemL1::GetLinearProgram(C.sparseView(), d, mu, &A, &b, &c);

  // Solve the linear program using the dual algorithm.
  linear_program_.Solve(A, b, c, ClpSolve::useDual, -1.0);

  // Extract the linear system solution y from the linear program solution x.
  y_.resize(C.cols());
  LinearSystemL1::GetYFromX(linear_program_.x(), &y_);

  // Compute dy/dC and dy/dd.
  basis_.Compute(linear_program_);
  DerivativeYWrtC(C);
  DerivativeYWrtD(d);
}

// In Section 3.2, y = y+ - y-, so dy/dC = dy+/dC - dy-/dC.  And in equations
// (9) and (10), the linear program solution x is [y+ y- t]^T, so we get
// dy+/dC and dy-/dC from dx/dC.
void LinearSystemL1Derivatives::DerivativeYWrtC(const MatrixXd &C) {
  derivative_y_wrt_C_.resize(C.cols(), C.rows() * C.cols());
  for (int i = 0; i < C.cols(); ++i) {
    for (int j = 0; j < C.rows(); ++j) {
      for (int k = 0; k < C.cols(); ++k) {
        const double derivative_y_plus_wrt_C =
            DerivativeXWrtC(C, i, j, k);
        const double derivative_y_minus_wrt_C =
            DerivativeXWrtC(C, i + C.cols(), j, k);
        derivative_y_wrt_C_(i, k * C.rows() + j) =
            derivative_y_plus_wrt_C - derivative_y_minus_wrt_C;
      }
    }
  }
}

// In equation (10), C appears in the linear program coefficient matrix A four
// times.  So, dx/dC is the sum of the four corresponding parts of dx/dA.
double LinearSystemL1Derivatives::DerivativeXWrtC(const MatrixXd &C,
                                                  int x_component,
                                                  int C_row,
                                                  int C_column) {
  return
      DerivativeXWrtA(x_component, C_row, C_column) -
      DerivativeXWrtA(x_component, C_row, C_column + C.cols()) -
      DerivativeXWrtA(x_component, C_row + C.rows(), C_column) +
      DerivativeXWrtA(x_component, C_row + C.rows(), C_column + C.cols());
}

// The linear program basis B is a subset of the columns of the coefficient
// matrix A.  Similarly, the basic solution x_B is a subset of
// the solution vector x, including the elements of x that correspond
// to the columns in the basis.  So, if x_i is present in x_B and
// A_{j, k} is present in B, then dx_i/dA_{j, k} is taken from from
// dx_B/dB, which is computed by DerivativeXBWrtB() below.
//
// Otherwise (if x_i is not in x_B or A_{j, k} is not in the basis),
// dx_i/dA_{j, k} is zero.
double LinearSystemL1Derivatives::DerivativeXWrtA(int x_component,
                                                  int A_row,
                                                  int A_column) {
  assert(x_component >= 0);
  assert(x_component < linear_program_.x().rows());
  assert(x_component < basis_.A_column_to_basis_column_map().size());
  assert(A_column <  basis_.A_column_to_basis_column_map().size());
  const int x_B_component = basis_.A_column_to_basis_column_map()[x_component],
      B_column = basis_.A_column_to_basis_column_map()[A_column];
  if (x_B_component != -1 && B_column != -1) {
    return DerivativeXBWrtB(x_B_component, A_row, B_column);
  } else {
    return 0;
  }
}

// dx_B/dB is given by equation (3) in the paper.  The chain of calls
// for computing dy/dC bottoms out here.
double LinearSystemL1Derivatives::DerivativeXBWrtB(int x_B_component,
                                                   int B_row,
                                                   int B_column) {
  return -basis_.x_B()(B_column) * basis_.B_inverse()(x_B_component, B_row);
}

// In Section 3.2, y = y+ - y-, so dy/dd = dy+/dd - dy-/dd.  And in equations
// (9) and (10), the linear program solution x is [y+ y- t]^T, so we get
// dy+/dd and dy-/dd from dx/dd.  This is similar to the reasoning for
// DerivativeYWrtC() above.
void LinearSystemL1Derivatives::DerivativeYWrtD(const VectorXd &d) {
  derivative_y_wrt_d_.resize(y_.rows(), d.rows());
  for (int i = 0; i < y_.rows(); ++i) {
    for (int j = 0; j < d.rows(); ++j) {
      derivative_y_wrt_d_(i, j) =
          DerivativeXWrtD(i, j, d.rows()) -
          DerivativeXWrtD(i + y_.rows(), j, d.rows());
    }
  }
}

// In equation (10), d appears in the linear program right-hand side b twice.
// So, we construct dx/dd from the two corresponding parts of dx/db.
double LinearSystemL1Derivatives::DerivativeXWrtD(int x_component,
                                                  int d_component,
                                                  int d_rows) {
  return
      DerivativeXWrtb(x_component, d_component) -
      DerivativeXWrtb(x_component, d_component + d_rows);
}

// The linear program's basic solution x_B is a subset of the solution vector
// x, including the elements of x that correspond to the columns in the basis.
// So, if x_i is present in x_B, then dx_i/db is taken from from dx_B/db,
// which is computed by DerivativeXBWrtb() below.  This is similar to
// the reasoning for DerivativeXWrtA() above.
//
// Otherwise (if x_i is not in x_B), dx_i/db is zero.
double LinearSystemL1Derivatives::DerivativeXWrtb(int x_component,
                                                  int b_component) {
  const int x_B_component = basis_.A_column_to_basis_column_map()[x_component];
  if (x_B_component != -1) {
    return DerivativeXBWrtb(x_B_component, b_component);
  } else {
    return 0;
  }
}

// dx_B/db is just B^{-1}, given by equation (4) in the paper.  The chain
// of calls for computing dy/dd bottoms out here.
double LinearSystemL1Derivatives::DerivativeXBWrtb(int x_B_component,
                                                   int b_component) {
  return basis_.B_inverse()(x_B_component, b_component);
}

}  // namespace wiberg
