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

#include "l1/linear_system/linear_system_l1.h"
#include <iostream>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SparseCore"
#include "l1/linear_system/linear_program.h"


using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

namespace wiberg {

void LinearSystemL1::Solve(const SparseMatrix<double> &C,
                           const VectorXd &d,
                           double mu,
                           ClpSolve::SolveType algorithm,
                           double time_limit,
                           VectorXd *y) {
  assert(y != NULL);

  // Convert the L1 minimization problem to a linear program and
  // solve it.
  SparseMatrix<double> A;
  VectorXd b, c;
  GetLinearProgram(C, d, mu, &A, &b, &c);
  LinearProgram linear_program;
  linear_program.Solve(A, b, c, algorithm, time_limit);

  // Extract the linear system solution from the linear program
  // solution.
  y->resize(C.cols());
  GetYFromX(linear_program.x(), y);
  double step_norm = y->lpNorm<1>();
  if (mu > 0.0 && step_norm > mu) {
    cout << "Warning: trust region step size " << step_norm
         << " > trust region " << mu << endl;
  }
}

void LinearSystemL1::GetLinearProgram(const SparseMatrix<double> &C,
                                      const VectorXd &d,
                                      double mu,
                                      SparseMatrix<double> *A,
                                      VectorXd *b,
                                      VectorXd *c) {
  assert(A != NULL);
  assert(b != NULL);
  assert(c != NULL);
  assert(C.rows() == d.rows());
  assert(mu >= 0.0);

  // Find the problem size.
  int out_rows = 2 * C.rows(), out_columns = 2 * C.cols() + d.rows();
  if (mu > 0.0) {
    out_rows++;
  }

  // Set c to be the objective vector in equation (9).
  (*c) = VectorXd::Zero(out_columns);
  for (int i = 2 * C.cols(); i < out_columns; ++i) {
    (*c)(i) = 1.0;
  }

  // Set b to be the right-hand side in equation (10).
  (*b) = VectorXd::Zero(out_rows);
  b->segment(0, d.rows()) = d;
  b->segment(d.rows(), d.rows()) = -d;

  // Set A to be the coefficient matrix in equation (10).
  int m = C.rows(), n = C.cols();
  A->resize(out_rows, out_columns);
  vector<Triplet<double> > triplets;
  for (int j = 0; j < C.outerSize(); ++j) {  // Iterate over columns.
    for (SparseMatrix<double>::InnerIterator it(C, j); it;
         ++it) {
      const int i = it.row();
      const double value = it.value();
      triplets.push_back(Triplet<double>(0 + i, 0 + j, value));
      triplets.push_back(Triplet<double>(0 + i, n + j, -value));
      triplets.push_back(Triplet<double>(m + i, 0 + j, -value));
      triplets.push_back(Triplet<double>(m + i, n + j, value));
    }
  }
  for (int i = 0; i < m; ++i) {
    triplets.push_back(Triplet<double>(0 + i, 2 * n + i, -1.0));
    triplets.push_back(Triplet<double>(m + i, 2 * n + i, -1.0));
  }

  // If there's a trust region constraint, add it as shown in the bottom
  // row of equation (14).
  if (mu > 0.0) {
    for (int i = 0; i < 2 * C.cols(); ++i) {
      triplets.push_back(Triplet<double>(A->rows() - 1, i, 1.0));
    }
    (*b)(A->rows() - 1) = mu;
  }
  A->setFromTriplets(triplets.begin(), triplets.end());
}

void LinearSystemL1::GetYFromX(const VectorXd &x, VectorXd *y) {
  assert(y != NULL);
  for (int i = 0; i < y->rows(); ++i) {
    (*y)(i) = x(i) - x(i + y->rows());
  }
}

}  // namespace wiberg
