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

// Factorization implements Wiberg L1 matrix factorization, originally
// described in Eriksson and van den Hengel, CVPR 2010.  Our implementation
// follows the clearer description in Strelow CVPR 2012, sections 3.3
// ("Nonlinear L1 Minimization") and 3.4 ("Wiberg L1 Factorization").
// There's a copy of the paper at research.google.com/pubs/pub37749.html.

#ifndef WIBERG_L1_FACTORIZATION_FACTOR_WIBERG_H_
#define WIBERG_L1_FACTORIZATION_FACTOR_WIBERG_H_

#include <vector>
#include "eigen3/Eigen/Dense"

namespace wiberg {

class ObservationToRowMap;
class SolveV;

class FactorWiberg {
 public:
  // Factors the matrix Y into U and V.
  //
  // Y can have missing entries.  If observation_mask(i, j) == 1.0, Y(i, j)
  // is an actual observation, otherwise Y(i, j) will be ignored.
  //
  // *U must be be an initial estimate for U.  Since the method solves
  // for V in closed form given U, any initial estimate for V in *V
  // is ignored.
  //
  // For num_iterations iterations, residuals will contain num_iterations + 1
  // residuals - the initial residual then the residual after each iteration.
  static void Factor(const Eigen::MatrixXd &Y,
                     const Eigen::MatrixXd &observation_mask,
                     int num_iterations,
                     Eigen::MatrixXd *U,
                     Eigen::MatrixXd *V,
                     std::vector<double> *residuals);

 private:
  // Implements equation (18) from the paper, which gives the total derivative
  // for a single prediction.
  static void GetTotalDerivativeSinglePredictionWrtU(
      int i,
      int u_rows,
      const Eigen::VectorXd &u_i,
      const Eigen::VectorXd &v_j,
      const Eigen::MatrixXd &derivative_v_wrt_U,
      Eigen::MatrixXd *derivative);
  // Computes the matrix of total derivatives for all of the predictions, by
  // collecting the results for the single predictions from equation (18) /
  // GetTotalDerivativeSinglePredictionWrtU() into one matrix.
  static void GetTotalDerivativePredictionsWrtU(
      const ObservationToRowMap &observation_to_row,
      const Eigen::MatrixXd &U,
      const SolveV &solve_V,
      Eigen::MatrixXd *derivative);
};

}  // namespace wiberg

#endif  // WIBERG_L1_FACTORIZATION_FACTOR_WIBERG_H_
