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

#include "l1/factorization/observation_to_row_map.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::make_pair;
using std::map;
using std::pair;

namespace wiberg {

void ObservationToRowMap::Set(const MatrixXd &observation_mask) {
  clear();
  int count = 0;
  for (int j = 0; j < observation_mask.cols(); ++j) {
    for (int i = 0; i < observation_mask.rows(); ++i) {
      if (observation_mask(i, j) == 1.0) {
        insert(make_pair(make_pair(i, j), count));
        ++count;
      }
    }
  }
}

void ObservationToRowMap::Flatten(const MatrixXd &in, VectorXd *out) const {
  assert(out != NULL);
  *out = VectorXd::Zero(size());
  for (ObservationToRowMap::const_iterator it = begin();
       it != end(); ++it) {
    const int in_row = it->first.first, in_column = it->first.second,
        out_row = it->second;
    assert(in_row >= 0);
    assert(in_row < in.rows());
    assert(in_column >= 0);
    assert(in_column < in.cols());
    assert(out_row >= 0);
    assert(out_row < out->rows());
    (*out)(out_row) = in(in_row, in_column);
  }
}

}  // namespace wiberg
