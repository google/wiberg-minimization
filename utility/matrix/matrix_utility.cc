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

#include "utility/matrix/matrix_utility.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace wiberg {

void MatrixUtility::Unflatten(const VectorXd &vector, int m, int n,
                              MatrixXd *matrix) {
  assert(matrix != NULL);
  assert(vector.rows() == m * n);
  *matrix = MatrixXd::Zero(m, n);
  int count = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      (*matrix)(j, i) = vector(count);
      ++count;
    }
  }
}

}  // namespace wiberg
