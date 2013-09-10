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

#include "l1/factorization/factor_wiberg.h"
#include <iostream>
#include "eigen3/Eigen/SparseCore"
#include "l1/factorization/observation_to_row_map.h"
#include "l1/factorization/solve_V.h"
#include "l1/factorization/solve_V_j.h"
#include "l1/linear_system/linear_system_l1.h"
#include "utility/matrix/matrix_utility.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

namespace wiberg {

void FactorWiberg::Factor(const MatrixXd &Y,
                          const MatrixXd &observation_mask,
                          int num_iterations,
                          MatrixXd *U,
                          MatrixXd *V,
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

  // Solve for V given the initial U.
  SolveV solve_V;
  solve_V.Solve(*U, Y, observation_mask);
  *V = solve_V.V();

  // Get the initial residuals.
  MatrixXd product_matrix, residuals_matrix;
  product_matrix = *U * solve_V.V();
  residuals_matrix = Y - product_matrix;
  VectorXd residuals_vector;
  observation_to_row_map.Flatten(residuals_matrix, &residuals_vector);
  double residual = residuals_vector.lpNorm<1>();
  (*residuals)[0] = residual;
  cout << "Initial residual: " << residual << endl;

  // Perform num_iterations Wiberg steps.  The initial trust region size
  // mu = 10 is arbitrary, but it will be adjusted on each iteration.
  double mu = 10.0;
  for (int i = 0; i < num_iterations; ++i) {
    // Get the derivative of the predictions with respect to U.
    MatrixXd derivative_predictions_wrt_U;
    GetTotalDerivativePredictionsWrtU(observation_to_row_map, *U, solve_V,
                                      &derivative_predictions_wrt_U);

    // Find the step.
    VectorXd step;
    LinearSystemL1::Solve(
        derivative_predictions_wrt_U.sparseView(),
        residuals_vector,
        mu,
        ClpSolve::usePrimal,
        -1.0,  // No time limit.
        &step);

    // Add the step to U and resolve for V.
    MatrixXd step_matrix;
    MatrixUtility::Unflatten(step, U->rows(), U->cols(), &step_matrix);
    const MatrixXd U_tentative = *U + step_matrix;
    SolveV solve_V_tentative;
    solve_V_tentative.Solve(U_tentative, Y, observation_mask);

    // Recompute the residuals.
    product_matrix = U_tentative * solve_V_tentative.V();
    const MatrixXd residuals_matrix_tentative = Y - product_matrix;
    VectorXd residuals_vector_tentative;
    observation_to_row_map.Flatten(residuals_matrix_tentative,
                                   &residuals_vector_tentative);
    const double residual_tentative = residuals_vector_tentative.lpNorm<1>();

    // If the new residual is smaller, accept the step and increase the trust
    // region size.  Otherwise, reject the step and try again on the next
    // iteration with a smaller trust region.
    if (residual_tentative < residual) {
      cout << "On iteration " << i
           << ", accepting step from residual " << residual
           << " to " << residual_tentative
           << "; mu goes to " << mu * 10 << endl;
      *U = U_tentative;
      *V = solve_V_tentative.V();
      solve_V.CopyFrom(solve_V_tentative);
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

void FactorWiberg::GetTotalDerivativeSinglePredictionWrtU(
    int i,
    int U_rows,
    const VectorXd &U_i,
    const VectorXd &V_j,
    const MatrixXd &derivative_V_j_wrt_U,
    MatrixXd *derivative) {
  assert(derivative != NULL);

  // This block implements the last term in equation (18).  The partial
  // derivative of the prediction with respect to V_j is just U_i.
  const MatrixXd partial_derivative_prediction_wrt_V_j = U_i.transpose();
  *derivative = partial_derivative_prediction_wrt_V_j * derivative_V_j_wrt_U;

  // This block adds in the middle term in equation (18).  As shown in equation
  // (19), the partial derivative of the prediction with respect to U_i is just
  // V_j.  That has to be added in the appropriate place for index i.
  for (int k = 0; k < V_j.rows(); ++k) {
    const int index = i + k * U_rows;
    (*derivative)(0, index) = (*derivative)(0, index) + V_j(k);
  }
}

void FactorWiberg::GetTotalDerivativePredictionsWrtU(
    const ObservationToRowMap &observation_to_row_map,
    const MatrixXd &U,
    const SolveV &solve_V,
    MatrixXd *derivative) {
  assert(derivative != NULL);
  *derivative = MatrixXd::Zero(observation_to_row_map.size(),
                               U.rows() * U.cols());

  // Add the derivative for each prediction.
  for (const auto &item : observation_to_row_map) {
    const int i = item.first.first, j = item.first.second, row = item.second;

    // Get the derivative for this prediction.
    VectorXd u_i = U.row(i);
    MatrixXd derivative_single_prediction;
    GetTotalDerivativeSinglePredictionWrtU(
        i,
        U.rows(),
        u_i,
        solve_V.solve_V_j(j).V_j(),
        solve_V.solve_V_j(j).derivative_V_j_wrt_U(),
        &derivative_single_prediction);
    assert(derivative_single_prediction.rows() == 1);
    assert(derivative_single_prediction.cols() == derivative->cols());

    // Put it in the appropriate row of the output matrix.
    for (int k = 0; k < derivative_single_prediction.cols(); ++k) {
      (*derivative)(row, k) = derivative_single_prediction(0, k);
    }
  }
}

}  // namespace wiberg
