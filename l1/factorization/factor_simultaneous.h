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

// FactorSimultaneous factors a matrix using successive linear programming,
// as described in section 3.5 ("Simultaneous L1 Factorization") of the paper,
// http://research.google.com/pubs/pub37749.html.  This method is competitive
// with Wiberg, providing a stronger baseline for comparison than EM, while also
// being much simpler than Wiberg.
//
// The "Simultaneous" part of FactorSimultaneous indicates that one linear
// system solve gives the updates for both U and V, rather than updating U
// only as in FactorWiberg.

#ifndef WIBERG_L1_FACTORIZATION_FACTOR_SIMULTANEOUS_H_
#define WIBERG_L1_FACTORIZATION_FACTOR_SIMULTANEOUS_H_

#include <vector>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SparseCore"

namespace wiberg {

class ObservationToRowMap;

class FactorSimultaneous {
 public:
  // Factors the matrix Y into U and V.  Most of the arguments are similar
  // to those in FactorWiberg::Factor():
  //
  //   Y can have missing entries.  If W_hat(i, j) == 1.0, Y(i, j) is an
  //   actual observation, otherwise Y(i, j) will be ignored.
  //
  //   *U and must be an initial estimate for U.  *V can be an initial estimate
  //   for V - see the new arguments below.
  //
  //   For num_iterations iterations, residuals will contain num_iterations + 1
  //   residuals - the initial residual then the residual after each iteration.
  //
  // The arguments that don't correspond to arguments in FactorWiberg::Factor()
  // are:
  //
  //   If initial_V_solve is true, then the initial estimate for V is found
  //   from U, Y, and observation_mask using the same approach as in the Wiberg
  //   Factor() method.  Otherwise, *V must contain an initial estimate.
  //   initial_V_solve = true provides the most direct comparison with Wiberg.
  //   But, since this finds a V based on the estimate of U, the initial
  //   estimate for V it finds can hurt performance if it is not as good
  //   as the supplied initial estimate.
  //
  //   If final_V_solve is true, then a final estimate for V is found from
  //   U, Y, and observation_mask after the final iteration, and used to compute
  //   the final residual.  As with initial_V_solve = true, this provides the
  //   most direct comparison with Wiberg, which also resolves for V on the
  //   final (actually every) iteration.
  static void Factor(const Eigen::MatrixXd &Y,
                     const Eigen::MatrixXd &observation_mask,
                     int num_iterations,
                     bool initial_V_solve,
                     bool final_V_solve,
                     Eigen::MatrixXd *U,
                     Eigen::MatrixXd *V,
                     std::vector<double> *residuals);

 private:
  // Computes the matrix of derivatives of the predictions, Y_{ij} = U_i * V_j,
  // with respect to U and V.
  static void GetDerivativePredictionsWrtUAndV(
      const ObservationToRowMap &observation_to_row_map,
      const Eigen::MatrixXd &U,
      const Eigen::MatrixXd &V,
      Eigen::SparseMatrix<double> *derivative);
};

}  // namespace wiberg

#endif  // WIBERG_L1_FACTORIZATION_FACTOR_SIMULTANEOUS_H_
