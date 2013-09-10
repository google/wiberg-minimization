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

#include <math.h>
#include <vector>
#include "l1/factorization/factor_simultaneous.h"
#include "eigen3/Eigen/Dense"

using Eigen::MatrixXd;
using std::vector;
using wiberg::FactorSimultaneous;

void TestFactorization() {
  // Create the test data.
  const int rows = 30, columns = 20, rank = 4;
  const MatrixXd
      U_ground_truth = MatrixXd::Random(rows, rank),
      V_ground_truth = MatrixXd::Random(rank, columns);
  MatrixXd
      Y = U_ground_truth * V_ground_truth,
      observation_mask = MatrixXd::Constant(rows, columns, 1.0),
      U = U_ground_truth + 0.1 * MatrixXd::Random(rows, rank),
      V = V_ground_truth + 0.1 * MatrixXd::Random(rank, columns);

  // Factor Y.
  vector<double> residuals, linear_programming_times;
  const int num_iterations = 10;
  FactorSimultaneous::Factor(Y, observation_mask, num_iterations,
                             true,  // initial_V_solve
                             true,  // final_V_solve
                             &U, &V,
                             &residuals);
}

int main(int argc, char *argv[]) {
  TestFactorization();
  return 0;
}
