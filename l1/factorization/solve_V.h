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

#ifndef WIBERG_L1_FACTORIZATION_SOLVE_V_H_
#define WIBERG_L1_FACTORIZATION_SOLVE_V_H_

#include <memory>
#include "eigen3/Eigen/Dense"

namespace wiberg {

class SolveVj;

// Given the observation matrix Y, the observation mask W_hat, and
// U, solves for V and dV/dU.
//
// Given U, the optimal solution for each column V_j of V can be found
// independently.  So, SolveV finds each V_j using an instance of SolveVj.
// For the derivative dV/dU, SolveV doesn't construct dV/dU explicity.
// Instead, dV/dU is stored across the SolveVj instances in their dV_j/dU's.
class SolveV {
 public:
  SolveV();
  ~SolveV();
  int size() const { return size_; }
  // Returns the solution for V_j, the jth column of V.
  const SolveVj &solve_V_j(int j) const;
  const Eigen::MatrixXd &V() const { return V_; }
  void CopyFrom(const SolveV &solve_V);
  void Solve(const Eigen::MatrixXd &U,
             const Eigen::MatrixXd &Y,
             const Eigen::MatrixXd &W_hat);
 private:
  int size_;
  std::unique_ptr<SolveVj[]> solve_V_j_;
  Eigen::MatrixXd V_;
};

}  // namespace wiberg

#endif  // WIBERG_L1_FACTORIZATION_SOLVE_V_H_
