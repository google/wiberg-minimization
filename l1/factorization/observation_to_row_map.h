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

// A map from entries in the observation matrix Y to positions in a flattened
// vector.  In many places we use use column-major arithmetic to flatten matrix
// elements' positions to vector positions, but Y can have missing entries
// whose positions should be excluded.

#ifndef WIBERG_L1_FACTORIZATION_OBSERVATION_TO_ROW_MAP_H_
#define WIBERG_L1_FACTORIZATION_OBSERVATION_TO_ROW_MAP_H_

#include <map>
#include <utility>
#include <vector>
#include "eigen3/Eigen/Dense"

namespace wiberg {

class ObservationToRowMap : public std::map<std::pair<int, int>, int> {
 public:
  ObservationToRowMap() {}
  ~ObservationToRowMap() {}
  // Builds the ObservationToRowMap from the observation mask.  If
  // observation_mask(i, j) == 1.0, Y(i, j) is an actual observation,
  // otherwise Y(i, j) will be excluded from the map.
  void Set(const Eigen::MatrixXd &observation_mask);
  // Flattens the observation matrix into a vector using the map.
  void Flatten(const Eigen::MatrixXd &in, Eigen::VectorXd *out) const;
};

}  // namespace wiberg

#endif  // WIBERG_L1_FACTORIZATION_OBSERVATION_TO_ROW_MAP_H_
