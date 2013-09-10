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

#include "l1/linear_system/linear_program.h"
#include <string>
#include <vector>
#include "coin/ClpMessage.hpp"
#include "coin/ClpSimplex.hpp"
#include "coin/CoinBuild.hpp"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;
using std::unique_ptr;

namespace wiberg {

LinearProgram::LinearProgram() {}

LinearProgram::~LinearProgram() {}

void LinearProgram::Solve(const SparseMatrix<double> &A,
                          const VectorXd &b,
                          const VectorXd &c,
                          ClpSolve::SolveType algorithm,
                          double time_limit) {
  // We can only set a time limit for the primal algorithm.
  assert(algorithm == ClpSolve::usePrimal || time_limit == -1.0);

  A_ = A;
  b_ = b;
  c_ = c;
  algorithm_ = algorithm;

  // Create the solver instance.
  solver_.reset(new ClpSimplex());
  solver_->setStrParam(ClpProbName, "x");
  solver_->setLogLevel(0);
  solver_->setMaximumSeconds(time_limit);
  ClpSolve options;
  options.setSolveType(algorithm);

  // Add the variables.
  const double kInfinity = std::numeric_limits<double>::infinity();
  solver_->resize(0, A.cols());
  for (int i = 0; i < A.cols(); ++i) {
    string name = "x_" + std::to_string(i);
    solver_->setColumnName(i, name);
    solver_->setColumnBounds(i,
                             0,  // Lower bound for this variable.
                             kInfinity);  // Upper bound for this variable.
  }

  // Add the constraints.  We want to iterate over A's columns in the inner
  // loop below using InnerIterator, which would require a row-major matrix,
  // but default SparseMatrix<double>'s are column-major.  So, we'll work
  // with a transposed matrix.  Even though we have to do the transpose it's
  // still a big win to be able to use the iterator over nonzero elements.
  CoinBuild build_object;
  unique_ptr<int[]> indices(new int[A.cols()]);
  unique_ptr<double[]> coefficients(new double[A.cols()]);
  SparseMatrix<double> A_transpose = A.transpose();
  for (int i = 0; i < A_transpose.outerSize(); ++i) {
    int count = 0;
    for (SparseMatrix<double>::InnerIterator it(A_transpose, i); it; ++it)  {
      // We want indices[count] to be the current column index of A, the
      // original untransposed matrix, and that will be the row index of
      // A_transpose.
      indices.get()[count] = it.row();
      coefficients.get()[count] = it.value();
      ++count;
    }
    build_object.addRow(count, indices.get(), coefficients.get(), -kInfinity,
                        b(i));
  }
  solver_->addRows(build_object);
  for (int i = 0; i < A.rows(); ++i) {
    string name = "constraint_" + std::to_string(i);
    solver_->setRowName(i, name);
  }

  // Add the objective.
  solver_->setOptimizationDirection(1);
  for (int i = 0; i < c.rows(); ++i) {
    solver_->setObjectiveCoefficient(i, c(i));
  }

  // Solve.
  solver_->initialSolve(options);

  // Extract the solution if feasible, else return zero.
  x_ = VectorXd::Zero(A.rows() + A.cols());
  if (solver_->status() == CLP_SIMPLEX_STOPPED ||  // The solution is feasible.
      solver_->status() == CLP_SIMPLEX_FINISHED) {  // The solution is optimal.
    const double* const values = solver_->getColSolution();
    for (int i = 0; i < A.cols(); ++i) {
      x_(i) = values[i];
    }
    const double* const row_activities = solver_->getRowActivity();
    for (int i = 0; i < A.rows(); ++i) {
      x_(A.cols() + i) = row_activities[i];
    }
  }
}

}  // namespace wiberg
