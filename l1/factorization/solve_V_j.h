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

#ifndef WIBERG_L1_FACTORIZATION_SOLVE_V_J_H_
#define WIBERG_L1_FACTORIZATION_SOLVE_V_J_H_

#include <vector>
#include "eigen3/Eigen/Dense"

namespace wiberg {

// Given the observation matrix Y, the observation mask, and
// U, solves for V_j (column j of V) and computes dV_j/dU.
class SolveVj {
 public:
  SolveVj() {}
  ~SolveVj() {}
  void CopyFrom(const SolveVj &solve_V_j);
  void Solve(const Eigen::MatrixXd &U,
             const Eigen::MatrixXd &Y,
             const Eigen::MatrixXd &observation_mask,
             int j);
  // Returns a vector of the indices of the observations present in Y's jth
  // column, found from the observation mask.
  const std::vector<int> &observed() const { return observed_; }
  const Eigen::VectorXd &V_j() const { return V_j_; }
  const Eigen::MatrixXd &derivative_V_j_wrt_U() const
    { return derivative_V_j_wrt_U_; }
 private:
  std::vector<int> observed_;
  Eigen::VectorXd V_j_;
  Eigen::MatrixXd derivative_V_j_wrt_U_;
};

}  // namespace wiberg

#endif  // WIBERG_L1_FACTORIZATION_SOLVE_V_J_H_
