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

#include "l1/factorization/factor_simultaneous.h"
#include <iostream>
#include "l1/factorization/observation_to_row_map.h"
#include "l1/factorization/solve_V.h"
#include "l1/linear_system/linear_system_l1.h"

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

namespace wiberg {

void FactorSimultaneous::Factor(const Eigen::MatrixXd &Y,
                                const Eigen::MatrixXd &observation_mask,
                                int num_iterations,
                                bool initial_V_solve,
                                bool final_V_solve,
                                Eigen::MatrixXd *U,
                                Eigen::MatrixXd *V,
                                vector<double> *residuals) {
  assert(U != NULL);
  assert(V != NULL);
  assert(residuals != NULL);

  // residuals will store the initial residual and the residual after each
  // iteration.
  *residuals = vector<double>(num_iterations + 1, 0.0);

  // Get the map that flattens the observation matrix entries to a vector.
  ObservationToRowMap observation_to_row_map;
  observation_to_row_map.Set(observation_mask);

  // If initial_V_solve, Solve for V given the initial U.
  if (initial_V_solve) {
    SolveV solve_V;
    solve_V.Solve(*U, Y, observation_mask);
    *V = solve_V.V();
  }

  // Get the initial residuals.
  MatrixXd product_matrix, residuals_matrix;
  product_matrix = *U * *V;
  residuals_matrix = Y - product_matrix;
  VectorXd residuals_vector;
  observation_to_row_map.Flatten(residuals_matrix, &residuals_vector);
  double residual = residuals_vector.lpNorm<1>();
  (*residuals)[0] = residual;
  cout << "Initial residual: " << residual << endl;

  // Perform num_iterations steps.  The initial trust region size mu = 10 is
  // arbitrary, but it will be adjusted on each iteration.
  double mu = 10.0;
  for (int i = 0; i < num_iterations; ++i) {
    // Get the derivative of the predictions with respect to U and V.
    SparseMatrix<double> derivative_predictions_wrt_U_V;
    GetDerivativePredictionsWrtUAndV(observation_to_row_map, *U, *V,
                                     &derivative_predictions_wrt_U_V);

    // Find the step.
    VectorXd step;
    LinearSystemL1::Solve(
        derivative_predictions_wrt_U_V,
        residuals_vector,
        mu,
        ClpSolve::usePrimal,
        -1.0,  // No time limit.
        &step);

    // Get the new U and V from the step.
    MatrixXd U_tentative(U->rows(), U->cols());
    for (int j = 0; j < U->rows(); ++j) {
      for (int k = 0; k < U->cols(); ++k) {
        U_tentative(j, k) = (*U)(j, k) + step(k * U->rows() + j);
      }
    }
    MatrixXd V_tentative(V->rows(), V->cols());
    int v_offset = U->rows() * U->cols();
    for (int j = 0; j < V->rows(); ++j) {
      for (int k = 0; k < V->cols(); ++k) {
        V_tentative(j, k) = (*V)(j, k) + step(v_offset + k * V->rows() + j);
      }
    }

    // Recompute the residuals.
    product_matrix = U_tentative * V_tentative;
    const MatrixXd residuals_matrix_tentative = Y - product_matrix;
    VectorXd residuals_vector_tentative;
    observation_to_row_map.Flatten(residuals_matrix_tentative,
                                   &residuals_vector_tentative);
    const double residual_tentative = residuals_vector_tentative.lpNorm<1>();

    // If the new residual is smaller, accept the step and increase the trust
    // region size.  Otherwise, reject the step and decrease the trust region
    // size.
    if (residual_tentative < residual) {
      cout << "On iteration " << i
           << ", accepting step from residual " << residual
           << " to " << residual_tentative
           << "; mu goes to " << mu * 10 << endl;
      *U = U_tentative;
      *V = V_tentative;
      residuals_vector = residuals_vector_tentative;
      residual = residual_tentative;
      mu *= 10.0;
    } else {
      cout << "On iteration " << i
           << ", rejecting step from residual " << residual
           << " to " << residual_tentative
           << "; mu goes to " << mu / 10 << endl;
      mu /= 10.0;
    }

    // Record the residual for this iteration.
    (*residuals)[i + 1] = residual;
  }
}

void FactorSimultaneous::GetDerivativePredictionsWrtUAndV(
    const ObservationToRowMap &observation_to_row,
    const MatrixXd &U,
    const MatrixXd &V,
    SparseMatrix<double> *derivative) {
  derivative->resize(observation_to_row.size(),  // Also zeroes the matrix.
                     U.rows() * U.cols() + V.rows() * V.cols());
  vector<Triplet<double>> triplets;
  for (const auto &item : observation_to_row) {
    const int i = item.first.first, j = item.first.second, row = item.second;

    // The derivative of the observation with respect to u_i is just v_j, and
    // vice versa, as shown in equations (21) and (22) in the paper.
    const VectorXd u_i = U.row(i), v_j = V.col(j);

    // Put the derivatives in the appropriate row of the output matrix.
    for (int k = 0; k < v_j.rows(); ++k) {
      const int v_offset = U.rows() * U.cols();
      triplets.push_back(
          Triplet<double>(row, k * U.rows() + i, v_j(k)));
      triplets.push_back(
          Triplet<double>(row, v_offset + j * V.rows() + k, u_i(k)));
    }
  }
  derivative->setFromTriplets(triplets.begin(), triplets.end());
}

}  // namespace research_vision_wiberg
