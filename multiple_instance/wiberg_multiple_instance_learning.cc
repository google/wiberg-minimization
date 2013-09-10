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

#include "multiple_instance/wiberg_multiple_instance_learning.h"
#include "eigen3/Eigen/SparseCore"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using std::vector;

namespace wiberg {

// Solve multiple instance learning problem using alternative optimization.
void WibergMultipleInstanceLearning::SolveAlternative() {
  double residual, old_objective;
  // Alternatively solve the two sets of variables w and v.
  for (int iter = 0; iter < max_iterations_; ++iter) {
    old_objective = objective_;
    // Step 1: fix v, solve w.
    SolveW();
    // Step 2: fix w, solve v.
    SolveV();
    // Calculate the residual and decide convergence.
    residual = old_objective - objective_;
    if (residual < epsilon_)
      return;
  }
}

// Solve multiple instance learning problem using simultaneous optimization.
void WibergMultipleInstanceLearning::SolveSimultaneous() {
  double residual, threshold = 0.05;
  // Model parameters from previous iteration.
  VectorXd old_w, old_v, old_w_plus, old_w_minus, old_error_positive,
    old_error_negative;
  double old_bias, old_objective;
  // When the solution approach local minimal, we force the solution
  // to be feasible. Otherwise, we allow continue from infeasible solution.
  bool force_feasible = false;
  SolveW();

  // Simultaneously optimize over w and v using successive linear program.
  for (int iter = 0; iter < max_iterations_; ++iter) {
    // Save the parameters from previous iteration.
    old_w = w_;
    old_v = v_;
    old_bias = bias_;
    old_objective = objective_;
    old_w_plus = w_plus_;
    old_w_minus = w_minus_;
    old_error_positive = error_positive_;
    old_error_negative = error_negative_;

    // Solve the LP by linearizing the constraints, save this middle solution
    // even it might be infeasible.
    SolveSimultaneousLinearProgram();
    double mid_objective = objective_;
    // Find the true objective by naive projection.
    ComputeFeasibleObjective();
    // Project the solution back into a feasible point by fixing v.
    SolveW();

    // Making decisions.
    // If forcing feasible, then we only accept feasible solution. Otherwise, we
    // allow infeasible solution.
    double delta1 = old_objective - mid_objective, delta2 = old_objective -
           true_objective_, delta3 = old_objective - objective_;
    if (force_feasible) {
      if (delta1 > 0.0 && delta3 > 0.0) {
        if (mu_ < 0.1)
          mu_ *= 10.0;
        // Decide convergence.
        if (delta3 < epsilon_)
          return;
      } else {
        w_ = old_w;
        v_ = old_v;
        bias_ = old_bias;
        w_plus_ = old_w_plus;
        w_minus_ = old_w_minus;
        objective_ = old_objective;
        error_positive_ = old_error_positive;
        error_negative_ = old_error_negative;
        mu_ /= 10.0;
      }
    } else if (delta2 > 0.0 || delta3 > 0.0) {
      if (delta1 > 0.0 && delta2 > 0.0 && delta3 > 0.0 &&
        delta1 / old_objective < threshold &&
        delta2 / old_objective < threshold &&
        delta3 / old_objective < threshold) {
        force_feasible = true;
        mu_ = 0.1;
      }
      if (!force_feasible && mu_ < 100.0 && delta2 > 0.0 && delta3 > 0.0)
        mu_ *= 10.0;
      if (!force_feasible && delta3 < 0.0)
        mu_ /= 10.0;
      // Calculate the residual and decide convergence.
      residual = objective_ - old_objective;
      if ( residual < 0.0)
        residual = -residual;
      if (residual < epsilon_)
        return;
    } else {
      w_ = old_w;
      v_ = old_v;
      bias_ = old_bias;
      w_plus_ = old_w_plus;
      w_minus_ = old_w_minus;
      objective_ = old_objective;
      error_positive_ = old_error_positive;
      error_negative_ = old_error_negative;
      mu_ /= 10.0;
    }
  }
}

// Solve multiple instance learning problem using wiberg optimization.
void WibergMultipleInstanceLearning::SolveWiberg() {
  double residual, threshold = 0.05;
  // Model parameters from previous iteration.
  VectorXd old_w, old_v, old_w_plus, old_w_minus, old_error_positive,
    old_error_negative;
  double old_bias, old_objective, old_true_objective;
  MatrixXd old_derivative_w_wrt_v;
  bool force_feasible = false;

  SolveW();
  GetDerivative_W_Wrt_V();

  // Wiberg optimization over v only using successive linear program.
  for (int iter = 0; iter < max_iterations_; iter++) {
    // Save the parameters from previous iteration.
    old_w = w_;
    old_v = v_;
    old_bias = bias_;
    old_objective = objective_;
    old_true_objective = true_objective_;
    old_w_plus = w_plus_;
    old_w_minus = w_minus_;
    old_error_positive = error_positive_;
    old_error_negative = error_negative_;
    old_derivative_w_wrt_v = derivative_w_wrt_v_;

    // Solve the wiberg LP by linearizing the objective and constraints.
    SolveWibergLinearProgram();
    double mid_objective = objective_;
    // Find the true objective by naive projection.
    ComputeFeasibleObjective();
    // Project the solution back into a feasible point by fixing v.
    SolveW();

    // If the new objective is decreasing, accept the
    // step and increase the trust region size. Otherwise, reject and decrease
    // the trust region size.
    double delta1 = old_objective - mid_objective, delta2 = old_objective -
           true_objective_, delta3 = old_objective - objective_;
    if (force_feasible) {
      if (delta1 > 0.0 && delta3 > 0.0) {
        if (mu_ < 1)
          mu_ *= 10.0;
        // Decide convergence.
        if (delta3 < epsilon_)
          return;
        GetDerivative_W_Wrt_V();
      } else {
        w_ = old_w;
        v_ = old_v;
        bias_ = old_bias;
        w_plus_ = old_w_plus;
        w_minus_ = old_w_minus;
        objective_ = old_objective;
        error_positive_ = old_error_positive;
        error_negative_ = old_error_negative;
        derivative_w_wrt_v_ = old_derivative_w_wrt_v;
        mu_ /= 10.0;
      }
    } else if (delta2 > 0.0 || delta3 > 0.0) {
      if (delta1 > 0.0 && delta2 > 0.0 && delta3 > 0.0
        && delta1 / old_objective < threshold
        && delta2 / old_objective < threshold
        && delta3 / old_objective < threshold) {
        force_feasible = true;
        mu_ = 0.1;
      }
      if (!force_feasible && mu_ < 100.0 && delta2 > 0.0 && delta3 > 0.0)
        mu_ *= 10.0;
      if (!force_feasible && delta3 < 0.0)
        mu_ /= 10.0;
      // Calculate the residual and decide convergence.
      residual = objective_ - old_objective;
      if ( residual < 0.0)
        residual = -residual;
      if (residual < epsilon_)
        return;
      GetDerivative_W_Wrt_V();
    } else {
      w_ = old_w;
      v_ = old_v;
      bias_ = old_bias;
      w_plus_ = old_w_plus;
      w_minus_ = old_w_minus;
      objective_ = old_objective;
      error_positive_ = old_error_positive;
      error_negative_ = old_error_negative;
      derivative_w_wrt_v_ = old_derivative_w_wrt_v;
      mu_ /= 10.0;
    }
  }
}

// Initialize the data and model parameters.
void WibergMultipleInstanceLearning::Init(
    const Eigen::MatrixXd &positive_bags,
    const Eigen::MatrixXd &negative_bags,
    const Eigen::VectorXd &positive_instances_per_bag,
    const Eigen::VectorXd &negative_instances_per_bag,
    double lambda1, double lambda2) {
  assert(lambda1 >= 0.0);
  assert(lambda2 >= 0.0);

  // Initialize multiple instance data.
  positive_bags_ = positive_bags;
  negative_bags_ = negative_bags;
  positive_instances_per_bag_ = positive_instances_per_bag;
  negative_instances_per_bag_ = negative_instances_per_bag;
  lambda1_ = lambda1;
  lambda2_ = lambda2;

  // Calculate the total number of instances.
  num_positive_instances_ = positive_instances_per_bag_.lpNorm<1>();
  num_negative_instances_ = negative_instances_per_bag_.lpNorm<1>();
  num_positive_bags_ = positive_instances_per_bag_.rows();
  num_negative_bags_ = negative_instances_per_bag_.rows();

  assert(num_positive_instances_ == positive_bags_.rows());
  assert(num_negative_instances_ == negative_bags_.rows());
  assert(positive_bags_.cols() == negative_bags_.cols());

  // Initialize model parameters.
  bias_ = 0.0;
  feature_dimension_ = positive_bags_.cols();
  w_ = VectorXd::Zero(feature_dimension_);
  w_plus_ = VectorXd::Zero(feature_dimension_);
  w_minus_ = VectorXd::Zero(feature_dimension_);
  v_.resize(num_positive_instances_);
  error_positive_ = VectorXd::Constant(num_positive_bags_, 1, 1.0);
  error_negative_ = VectorXd::Constant(num_negative_instances_, 1, 1.0);
  derivative_w_wrt_v_.resize(2 * feature_dimension_, num_positive_instances_ -
    num_positive_bags_);
  mu_ = 10.0;
  max_iterations_ = 40;
  epsilon_ = 0.0001;

  // Initialize the weight parameter v to be equal.
  int count = 0;
  for (int i = 0; i < num_positive_bags_; ++i) {
    for (int j = 0; j < positive_instances_per_bag_(i); ++j) {
      v_(count) = 1.0 / positive_instances_per_bag_(i);
      ++count;
    }
  }

  // Init Objective
  ComputeObjective();
}

// Solve for optimal w by fixing v.
void WibergMultipleInstanceLearning::SolveW() {
  // Calculate average positive instance for each positive bag.
  MatrixXd average_positive_instance = CalculateAveragePositiveInstance();

  SparseMatrix<double> A;
  VectorXd b, c;

  //  Find problem output size.
  int out_rows = num_positive_bags_ + num_negative_instances_;
  int out_cols = 2 * feature_dimension_ + num_positive_bags_ +
      num_negative_instances_ + 2;

  int offset_error_positive = 2 * feature_dimension_;
  int offset_error_negative = offset_error_positive + num_positive_bags_;
  int offset_bias = offset_error_negative + num_negative_instances_;

  // Set c to be the objective vector.
  // We want to minimize the L1 norm of w together with
  // positive and negative errors.
  // objective: |w| + lambda1 * error_positive + lambda2 * error_negative
  c.resize(out_cols);
  c.segment(0, offset_error_positive) =
    VectorXd::Constant(offset_error_positive, 1, 1.0);
  c.segment(offset_error_positive, num_positive_bags_) =
    VectorXd::Constant(num_positive_bags_, 1, lambda1_);
  c.segment(offset_error_negative, num_negative_instances_) =
    VectorXd::Constant(num_negative_instances_, 1, lambda2_);
  c(offset_bias) = 0.0;
  c(offset_bias + 1) = 0.0;

  // Set A to be the coefficient matrix.
  // A is composed of two sets of constraints. Constraints by the positive
  // bags and by the negative instances, which are the standard constraints
  // as imposed in SVM models.
  vector<Triplet<double>> triplets;
  A.resize(out_rows, out_cols);
  for (int i = 0; i < num_positive_bags_; ++i) {
    for (int k = 0; k < feature_dimension_; ++k) {
      triplets.push_back(Triplet<double>(i, k,
                                         -average_positive_instance(i, k)));
      triplets.push_back(Triplet<double>(i, feature_dimension_ + k,
                                         average_positive_instance(i, k)));
    }
    triplets.push_back(Triplet<double>(i, offset_error_positive + i, -1.0));
    triplets.push_back(Triplet<double>(i, offset_bias, -1.0));
    triplets.push_back(Triplet<double>(i, offset_bias + 1, 1.0));
  }
  for (int i = 0; i < num_negative_instances_; ++i) {
    for (int k = 0; k < negative_bags_.cols(); ++k) {
      triplets.push_back(Triplet<double>(i + num_positive_bags_,
                                         k, negative_bags_(i, k)));
      triplets.push_back(Triplet<double>(i + num_positive_bags_,
                             negative_bags_.cols() + k, -negative_bags_(i, k)));
    }
    triplets.push_back(Triplet<double>(i + num_positive_bags_,
                                       offset_error_negative + i, -1.0));
    triplets.push_back(Triplet<double>(i + num_positive_bags_,
                                       offset_bias, 1.0));
    triplets.push_back(Triplet<double>(i + num_positive_bags_,
                                       offset_bias + 1, -1.0));
  }
  A.setFromTriplets(triplets.begin(), triplets.end());

  //  Set b to be the right-hand side.
  b.resize(out_rows);
  b.segment(0, out_rows) = VectorXd::Constant(out_rows, 1, -1.0);

  // Solve the linear program.
  linear_program_.Solve(A, b, c, ClpSolve::useDual, -1.0);

  // Extract solution from the linear program.
  ExtractWfromLinearProgram(linear_program_.x());

  // Compute the objective
  ComputeObjective();
}

// Solve for optimal v by fixing w.
void WibergMultipleInstanceLearning::SolveV() {
  // Calculate the dot product of w and instances.
  VectorXd positive_dot_product = positive_bags_ * w_,
    negative_dot_product = negative_bags_ * w_;

  SparseMatrix<double> A;
  VectorXd b, c;

  // Find problem sizes for two steps.
  int out_rows = 2 * num_positive_bags_ + num_negative_instances_;
  int out_cols = num_positive_instances_ - num_positive_bags_ +
      num_positive_bags_ + num_negative_instances_ + 2;

  int offset_error_positive = num_positive_instances_ - num_positive_bags_;
  int offset_error_negative = offset_error_positive + num_positive_bags_;
  int offset_bias = offset_error_negative +
      num_negative_instances_;

  //  Set c to be the objective vector.
  c.resize(out_cols);
  c.segment(0, offset_error_positive) =
    VectorXd::Constant(offset_error_positive, 1, 0.0);
  c.segment(offset_error_positive, num_positive_bags_) =
    VectorXd::Constant(num_positive_bags_, 1, lambda1_);
  c.segment(offset_error_negative, num_negative_instances_) =
    VectorXd::Constant(num_negative_instances_, 1, lambda2_);
  c(offset_bias) = 0.0;
  c(offset_bias + 1) = 0.0;

  //  Set A to be the coefficient matrix, based on
  //  modification of equality constraint to inequality.
  vector<Triplet<double>> triplets;
  A.resize(out_rows, out_cols);
  int count = 0;
  int index = 0;
  for (int i = 0; i < num_positive_bags_; i++) {
    for (int j = 0; j < positive_instances_per_bag_(i) - 1; j++) {
      triplets.push_back(Triplet<double>(
          i, count + j,
          positive_dot_product(index) - positive_dot_product(index + j + 1)));
      triplets.push_back(Triplet<double>(i + num_positive_bags_,
                                         count + j, 1.0));
    }
    count += positive_instances_per_bag_(i) - 1;
    index += positive_instances_per_bag_(i);
    triplets.push_back(Triplet<double>(i, offset_error_positive + i, -1.0));
    triplets.push_back(Triplet<double>(i, offset_bias, 1.0));
    triplets.push_back(Triplet<double>(i, offset_bias + 1, 1.0));
  }
  for (int i = 0; i < num_negative_instances_; i++) {
    triplets.push_back(Triplet<double>(i + 2 * num_positive_bags_,
                                       offset_error_negative + i, -1.0));
    triplets.push_back(Triplet<double>(i + 2 * num_positive_bags_,
                                       offset_bias, -1.0));
    triplets.push_back(Triplet<double>(i + 2 * num_positive_bags_,
                                       offset_bias + 1, -1.0));
  }
  A.setFromTriplets(triplets.begin(), triplets.end());

  //  Set b to be the right-hand side.
  b.resize(out_rows);
  index = 0;
  for (int i = 0; i < num_positive_bags_; i++) {
    b(i) = -1.0 + positive_dot_product(index);
    b(i + num_positive_bags_) = 1.0;
    index += positive_instances_per_bag_(i);
  }
  for (int i = 0; i < num_negative_instances_; i++) {
    b(i + 2 * num_positive_bags_) =
        -1.0 - negative_dot_product(i);
  }

  // Solve the linear program.
  linear_program_.Solve(A, b, c, ClpSolve::useDual, -1.0);

  // Extract solution from the linear program.
  ExtractVfromLinearProgram(linear_program_.x());

  // Calculate the objective value.
  ComputeObjective();
}

void WibergMultipleInstanceLearning::SolveSimultaneousLinearProgram() {
  // Calculate average positive instance for each positive bag.
  MatrixXd average_positive_instance = CalculateAveragePositiveInstance();

  // Calculate the positive bag prediction, w'B'v+bias,
  // negative instance prediction, w'B'+bias and positive dot product, w'B'
  VectorXd positive_bag_prediction, negative_instance_prediction,
    positive_dot_product;
  positive_bag_prediction = average_positive_instance * w_ +
    VectorXd::Constant(num_positive_bags_, 1, bias_);
  negative_instance_prediction = negative_bags_ * w_ +
    VectorXd::Constant(num_negative_instances_, 1, bias_);
  positive_dot_product = positive_bags_ * w_;

  // Define linear program variables.
  SparseMatrix<double> A;
  VectorXd b, c;

  // Find problem output size.
  int offset_v = 2 * feature_dimension_;
  int offset_bias = offset_v + num_positive_instances_ - num_positive_bags_;
  int offset_error_positive = offset_bias + 1;
  int offset_error_negative = offset_error_positive + num_positive_bags_;
  int variable_offset = offset_error_negative + num_negative_instances_;

  // Find number of variables and constraints.
  int out_rows = offset_bias + num_positive_bags_ + 2 * (num_positive_bags_ +
    num_negative_instances_) + 2 * variable_offset;
  int out_cols = 2 * variable_offset;

  // We want to minimize the objective: |w| + lambda1 * err+ + lambda2 * err-
  // with constraints: w'B'v + b >= 1-err+, -w'B' - b >= 1 - err-, sum_v = 1
  // using successive linear program by linearizing the constraints.
  // The variables are :
  // [dw+,dw-,dv,db,derr+,derr-]+ and [dw+,dw-,dv,db,derr+,derr-]-.
  // Set c to be the linear approximation of the objective.
  c = VectorXd::Zero(out_cols);
  c.segment(0, offset_v) = VectorXd::Constant(offset_v, 1, 1.0);
  c.segment(offset_error_positive, num_positive_bags_) =
    VectorXd::Constant(num_positive_bags_, 1, lambda1_);
  c.segment(offset_error_negative, num_negative_instances_) =
    VectorXd::Constant(num_negative_instances_, 1, lambda2_);
  c.segment(variable_offset, variable_offset) = -c.segment(0, variable_offset);

  // Set A to be the coefficient matrix which is composed of constraints:
  // w+ + dw+ >= 0, w- + dw- >= 0, v + dv >= 0, sum_(v + dv) <= 1
  // w'B'v + b - 1 + err+ + dw'B'v + w'B'dv + db + derr+ >= 0
  // -w'B'- b - 1 + err- - dw'B' - db + derr- >= 0
  // err(+-) + derr(+-) >= 0 and step <= mu
  vector<Triplet<double>> triplets;
  int offset = 0, count = 0, index =0;
  A.resize(out_rows, out_cols);
  // Set constraints: w+ + dw+ >= 0, w- + dw- >= 0, v + dv >= 0
  for (int i = 0; i < offset_bias; ++i) {
    triplets.push_back(Triplet<double>(offset + i, i, -1.0));
    triplets.push_back(Triplet<double>(offset + i, variable_offset + i, 1.0));
  }
  // Set constraints: sum_(v + dv) <= 1
  offset += offset_bias;
  count = offset_v;
  for (int i = 0; i < num_positive_bags_; ++i) {
    for (int j = 0; j < positive_instances_per_bag_(i) - 1; ++j) {
      triplets.push_back(Triplet<double>(offset + i, count + j, 1.0));
      triplets.push_back(Triplet<double>(offset + i,
               variable_offset + count + j, -1.0));
    }
    count += positive_instances_per_bag_(i) - 1;
  }
  // Set constraints: w'B'v + b - 1 + err+ + dw'B'v + w'B'dv + db + derr+ >= 0
  offset += num_positive_bags_;
  index = 0;
  count = offset_v;
  for (int i = 0; i < num_positive_bags_; ++i) {
    for (int j = 0; j < feature_dimension_; ++j) {
      triplets.push_back(Triplet<double>(offset + i, j,
               -average_positive_instance(i, j)));
      triplets.push_back(Triplet<double>(offset + i, variable_offset + j,
               average_positive_instance(i, j)));
      triplets.push_back(Triplet<double>(offset + i, feature_dimension_ + j,
               average_positive_instance(i, j)));
      triplets.push_back(Triplet<double>(offset + i, variable_offset +
               feature_dimension_ + j, -average_positive_instance(i, j)));
    }
    for (int j = 0; j < positive_instances_per_bag_(i) - 1; ++j) {
      double value = positive_dot_product(index) -
               positive_dot_product(index + j + 1);
      triplets.push_back(Triplet<double>(offset + i, count + j, value));
      triplets.push_back(Triplet<double>(offset + i,
               variable_offset + count + j, -value));
    }
    count += positive_instances_per_bag_(i) - 1;
    index += positive_instances_per_bag_(i);
    triplets.push_back(Triplet<double>(offset + i, offset_bias, -1.0));
    triplets.push_back(Triplet<double>(offset + i,
               variable_offset + offset_bias, 1.0));
    triplets.push_back(Triplet<double>(offset + i,
               offset_error_positive + i, -1.0));
    triplets.push_back(Triplet<double>(offset + i,
               variable_offset + offset_error_positive + i, 1.0));
  }
  // Set Constraints: -w'B'- b - 1 + err- - dw'B' - db + derr- >= 0
  offset += num_positive_bags_;
  for (int i = 0; i < num_negative_instances_; ++i) {
    for (int j = 0; j < feature_dimension_; ++j) {
      triplets.push_back(Triplet<double>(offset + i, j,
               negative_bags_(i, j)));
      triplets.push_back(Triplet<double>(offset + i, variable_offset + j,
               -negative_bags_(i, j)));
      triplets.push_back(Triplet<double>(offset + i, feature_dimension_ + j,
               -negative_bags_(i, j)));
      triplets.push_back(Triplet<double>(offset + i, variable_offset +
               feature_dimension_ + j, negative_bags_(i, j)));
    }
    triplets.push_back(Triplet<double>(offset + i, offset_bias, 1.0));
    triplets.push_back(Triplet<double>(offset + i,
               variable_offset + offset_bias, -1.0));
    triplets.push_back(Triplet<double>(offset + i,
               offset_error_negative + i, -1.0));
    triplets.push_back(Triplet<double>(offset + i,
               variable_offset + offset_error_negative + i, 1.0));
  }
  // Set constraints: err(+-) + derr(+-) >= 0
  offset += num_negative_instances_;
  for (int i = 0; i < num_positive_bags_ + num_negative_instances_; ++i) {
    triplets.push_back(Triplet<double>(offset + i,
               offset_error_positive + i, -1.0));
    triplets.push_back(Triplet<double>(offset + i,
               variable_offset + offset_error_positive + i, 1.0));
  }
  // Set constraints: step <= mu
  offset += num_positive_bags_ + num_negative_instances_;
  for (int i = 0; i < out_cols; ++i) {
    triplets.push_back(Triplet<double>(offset + i, i, 1.0));
  }
  A.setFromTriplets(triplets.begin(), triplets.end());

  // Set b to be the right-hand side.
  b.resize(out_rows);
  // Set w+ >= 0, w- >= 0, v >= 0 sum_v <= 1.
  b.segment(0, feature_dimension_) = w_plus_;
  b.segment(feature_dimension_, feature_dimension_) = w_minus_;
  index = 0; count = 0;
  for (int i = 0; i < num_positive_bags_; ++i) {
    b.segment(offset_v + count, positive_instances_per_bag_(i) - 1) =
      v_.segment(index + 1, positive_instances_per_bag_(i) - 1);
    b(offset_bias + i) = v_(index);
    count += positive_instances_per_bag_(i) - 1;
    index += positive_instances_per_bag_(i);
  }
  // Set w'B'v + b >= 1 - err+.
  offset = offset_bias + num_positive_bags_;
  b.segment(offset, num_positive_bags_) = positive_bag_prediction +
    error_positive_ - VectorXd::Constant(num_positive_bags_, 1, 1.0);
  // Set -w'B' - b >= 1 - err-.
  offset += num_positive_bags_;
  b.segment(offset, num_negative_instances_) = -negative_instance_prediction +
    error_negative_ - VectorXd::Constant(num_negative_instances_, 1, 1.0);
  // Set err >= 0.
  offset += num_negative_instances_;
  b.segment(offset, num_positive_bags_) = error_positive_;
  offset += num_positive_bags_;
  b.segment(offset, num_negative_instances_) = error_negative_;
  // Set step <= mu.
  offset += num_negative_instances_;
  b.segment(offset, out_cols) = VectorXd::Constant(out_cols, 1, mu_);

  // Solve the linear program.
  linear_program_.Solve(A, b, c, ClpSolve::useDual, -1.0);

  // Extract solution from the linear program.
  ExtractSimultaneousSolutionfromLinearProgram(linear_program_.x());

  // Calculate the objective value.
  ComputeObjective();
}

void WibergMultipleInstanceLearning::SolveWibergLinearProgram() {
  // Calculate average positive instance for each positive bag.
  MatrixXd average_positive_instance = CalculateAveragePositiveInstance();

  // Calculate the positive bag prediction, w'B'v + bias,
  // negative instance prediction, w'B' + bias and positive dot product, w'B'
  VectorXd positive_bag_prediction, negative_instance_prediction,
    positive_dot_product;
  positive_bag_prediction = average_positive_instance * w_ +
    VectorXd::Constant(num_positive_bags_, 1, bias_);
  negative_instance_prediction = negative_bags_ * w_ +
    VectorXd::Constant(num_negative_instances_, 1, bias_);
  positive_dot_product = positive_bags_ * w_;

  // Define linear program variables.
  SparseMatrix<double> A;
  VectorXd b, c;

  // Find problem output size.
  // Variables are [dv, db, derr+, derr-]+ and [dv, db, derr+, derr-]-.
  // Constraints are: v >= 0 sum_v <= 1 w'B'v + b >= 1 - err+
  // -w'B' - b >= 1 - err- and step <= mu
  int offset_bias = num_positive_instances_ - num_positive_bags_;
  int offset_error_positive = offset_bias + 1;
  int offset_error_negative = offset_error_positive + num_positive_bags_;
  int variable_offset = offset_error_negative + num_negative_instances_;

  // Find number of variables and constraints.
  int out_rows = offset_bias + num_positive_bags_ + 2 * (num_positive_bags_ +
    num_negative_instances_) + 2 * variable_offset + 2 * feature_dimension_;
  int out_cols = 2 * variable_offset;

  // We want to minimize the objective: |w(v)| + lambda1 * err+ + lambda2 * err-
  // with constraints: w'B'v + b >= 1 - err+, -w'B' - b >= 1 - err- and
  // sum_v <= 1 using successive linear program by linearizing the objective
  // and constraints.
  // Set c to be the gradient of linear approximation of the objective.
  c = VectorXd::Zero(out_cols);
  for (int i = 0; i < 2 * feature_dimension_; i++) {
    c.segment(0, offset_bias) += derivative_w_wrt_v_.row(i);
  }
  c.segment(offset_error_positive, num_positive_bags_) =
    VectorXd::Constant(num_positive_bags_, 1, lambda1_);
  c.segment(offset_error_negative, num_negative_instances_) =
    VectorXd::Constant(num_negative_instances_, 1, lambda2_);
  c.segment(variable_offset, variable_offset) = -c.segment(0, variable_offset);

  // Set A to be the coefficient matrix which is composed of the constraints.
  vector<Triplet<double>> triplets;
  int offset = 0, count = 0, index = 0;
  A.resize(out_rows, out_cols);
  // Set constraints: v >= 0.
  for (int i = 0; i < offset_bias; i++) {
    triplets.push_back(Triplet<double>(offset + i, i, -1.0));
    triplets.push_back(Triplet<double>(offset + i, variable_offset + i, 1.0));
  }
  // Set constraints: sum_v <= 1.
  offset += offset_bias;
  count = 0;
  for (int i = 0; i < num_positive_bags_; i++) {
    for (int j = 0; j < positive_instances_per_bag_(i) - 1; j++) {
      triplets.push_back(Triplet<double>(offset + i, count + j, 1.0));
      triplets.push_back(Triplet<double>(offset + i,
               variable_offset + count + j, -1.0));
    }
    count += positive_instances_per_bag_(i) - 1;
  }
  // Set constraints: w'B'v + b >= 1 - err+.
  offset += num_positive_bags_;
  index = 0; count = 0;
  for (int i = 0; i < num_positive_bags_; i++) {
    // Compute the derivative of positive constraint i wrt v,
    // which is -dw/dv B'v - w'B'.
    VectorXd derivative_positive_i_wrt_v = VectorXd::Zero(offset_bias);
    for (int j = 0; j < offset_bias; j++) {
      VectorXd derivative_j = derivative_w_wrt_v_.col(j).segment(
        feature_dimension_, feature_dimension_) -
        derivative_w_wrt_v_.col(j).segment(0, feature_dimension_);
      derivative_positive_i_wrt_v(j) += derivative_j.dot(
        average_positive_instance.row(i));
    }
    for (int j = 0; j < positive_instances_per_bag_(i) - 1; j++) {
      double value = positive_dot_product(index) -
               positive_dot_product(index + j + 1);
      derivative_positive_i_wrt_v(count + j) += value;
    }
    for (int j = 0; j < offset_bias; j++) {
      triplets.push_back(Triplet<double>(offset + i, j,
        derivative_positive_i_wrt_v(j)));
      triplets.push_back(Triplet<double>(offset + i, variable_offset + j,
        -derivative_positive_i_wrt_v(j)));
    }
    count += positive_instances_per_bag_(i) - 1;
    index += positive_instances_per_bag_(i);
    triplets.push_back(Triplet<double>(offset + i, offset_bias, -1.0));
    triplets.push_back(Triplet<double>(offset + i,
               variable_offset + offset_bias, 1.0));
    triplets.push_back(Triplet<double>(offset + i,
               offset_error_positive + i, -1.0));
    triplets.push_back(Triplet<double>(offset + i,
               variable_offset + offset_error_positive + i, 1.0));
  }
  // Set constraints: -w'B' - b >= 1 - err-.
  offset += num_positive_bags_;
  for (int i = 0; i < num_negative_instances_; i++) {
    // Compute the derivative of negative constraint i wrt v,
    // which is dw/dv B.
    VectorXd derivative_negative_i_wrt_v = VectorXd::Zero(offset_bias);
    for (int j = 0; j < offset_bias; j++) {
      VectorXd derivative_j = derivative_w_wrt_v_.col(j).segment(0,
        feature_dimension_) - derivative_w_wrt_v_.col(j).segment(
          feature_dimension_, feature_dimension_);
      derivative_negative_i_wrt_v(j) += derivative_j.dot(negative_bags_.row(i));
      triplets.push_back(Triplet<double>(offset + i, j,
        derivative_negative_i_wrt_v(j)));
      triplets.push_back(Triplet<double>(offset + i, variable_offset + j,
        -derivative_negative_i_wrt_v(j)));
    }
    triplets.push_back(Triplet<double>(offset + i, offset_bias, 1.0));
    triplets.push_back(Triplet<double>(offset + i,
               variable_offset + offset_bias, -1.0));
    triplets.push_back(Triplet<double>(offset + i,
               offset_error_negative + i, -1.0));
    triplets.push_back(Triplet<double>(offset + i,
               variable_offset + offset_error_negative + i, 1.0));
  }
  // Set constraints: err >= 0.
  offset += num_negative_instances_;
  for (int i = 0; i < num_positive_bags_ + num_negative_instances_; i++) {
    triplets.push_back(Triplet<double>(offset + i,
               offset_error_positive + i, -1.0));
    triplets.push_back(Triplet<double>(offset + i,
               variable_offset + offset_error_positive + i, 1.0));
  }
  // Set constraints: step <= mu.
  offset += num_positive_bags_ + num_negative_instances_;
  for (int i = 0; i < out_cols; i++) {
    triplets.push_back(Triplet<double>(offset + i, i, 1.0));
  }
  // Set constraints: w+ >= 0 and w- >= 0.
  offset += out_cols;
  for (int i = 0; i < 2 * feature_dimension_; i++) {
    for (int j = 0; j < offset_bias; j++) {
      triplets.push_back(Triplet<double>(offset + i, j,
        -derivative_w_wrt_v_(i, j)));
      triplets.push_back(Triplet<double>(offset + i, variable_offset + j,
        derivative_w_wrt_v_(i, j)));
    }
  }
  A.setFromTriplets(triplets.begin(), triplets.end());

  // Set b to be the right-hand side.
  b.resize(out_rows);
  // Set v >= 0 and sum_v <= 1.
  index = 0; count = 0;
  for (int i = 0; i < num_positive_bags_; i++) {
    b.segment(count, positive_instances_per_bag_(i) - 1) =
      v_.segment(index + 1, positive_instances_per_bag_(i) - 1);
    b(offset_bias + i) = v_(index);
    count += positive_instances_per_bag_(i) - 1;
    index += positive_instances_per_bag_(i);
  }
  // Set w'B'v + b >= 1 - err+.
  offset = offset_bias + num_positive_bags_;
  b.segment(offset, num_positive_bags_) = positive_bag_prediction +
    error_positive_ - VectorXd::Constant(num_positive_bags_, 1, 1.0);
  // Set -w'B' - b >= 1 - err-.
  offset += num_positive_bags_;
  b.segment(offset, num_negative_instances_) = -negative_instance_prediction +
    error_negative_ - VectorXd::Constant(num_negative_instances_, 1, 1.0);
  // Set err >= 0.
  offset += num_negative_instances_;
  b.segment(offset, num_positive_bags_) = error_positive_;
  offset += num_positive_bags_;
  b.segment(offset, num_negative_instances_) = error_negative_;
  // Set step <= mu.
  offset += num_negative_instances_;
  b.segment(offset, out_cols) = VectorXd::Constant(out_cols, 1, mu_);
  // Set w+ >= 0 and w- >= 0.
  offset += out_cols;
  b.segment(offset, feature_dimension_) = w_plus_;
  b.segment(offset + feature_dimension_, feature_dimension_) = w_minus_;

  // Solve the linear program.
  linear_program_.Solve(A, b, c, ClpSolve::useDual, -1.0);

  // Extract solution from the linear program.
  ExtractWibergSolutionfromLinearProgram(linear_program_.x());

  // Compute Objective.
  ComputeObjective();
}

void WibergMultipleInstanceLearning::ExtractWfromLinearProgram
(const Eigen::VectorXd &x) {
  int offset_error_positive = 2 * feature_dimension_;
  int offset_error_negative = offset_error_positive + num_positive_bags_;
  int offset_bias = offset_error_negative + num_negative_instances_;
  // x is of the form [w+  w- error_positive error_negative bias+ bias-]
  w_plus_.segment(0, feature_dimension_) = x.segment(0, feature_dimension_);
  w_minus_.segment(0, feature_dimension_) = x.segment(feature_dimension_,
    feature_dimension_);
  w_ = w_plus_ - w_minus_;
  error_positive_ = x.segment(offset_error_positive, num_positive_bags_);
  error_negative_ = x.segment(offset_error_negative, num_negative_instances_);
  bias_ = x(offset_bias) - x(offset_bias + 1);
}

void WibergMultipleInstanceLearning::ExtractVfromLinearProgram
(const Eigen::VectorXd &x) {
  int offset_error_positive = num_positive_instances_ - num_positive_bags_;
  int offset_error_negative = offset_error_positive + num_positive_bags_;
  int offset_bias = offset_error_negative + num_negative_instances_;
  // x is of the form [v error_positive error_negative bias+ bias-]
  // Note that here v is of length num_positive_instances_ -
  // num_positive_bags_, since we modify the equality
  // constraints into inequality and thus we have
  // num_positive_bags_ less variables.
  int count = 0, index = 0;
  for (int i = 0; i < num_positive_bags_; i++) {
    int num_instance = positive_instances_per_bag_(i);
    v_.segment(index + 1, num_instance - 1) =
      x.segment(count, num_instance - 1);
    v_(index) = 1.0 - v_.segment(index + 1, num_instance - 1).lpNorm<1>();
    count += num_instance - 1;
    index += num_instance;
  }
  error_positive_ = x.segment(offset_error_positive, num_positive_bags_);
  error_negative_ = x.segment(offset_error_negative, num_negative_instances_);
  bias_ = x(offset_bias) - x(offset_bias + 1);
}

void WibergMultipleInstanceLearning::
ExtractSimultaneousSolutionfromLinearProgram(const Eigen::VectorXd &x) {
  int offset_v = 2 * feature_dimension_;
  int offset_bias = offset_v + num_positive_instances_ - num_positive_bags_;
  int offset_error_positive = offset_bias + 1;
  int offset_error_negative = offset_error_positive + num_positive_bags_;
  int variable_offset = offset_error_negative + num_negative_instances_;
  // x is of the form [dw+ dw-+ dv+ db+ derr+ derr-]+ together with
  // [dw+ dw-+ dv+ db+ derr+ derr-]-.
  w_plus_ += x.segment(0, feature_dimension_) -
    x.segment(variable_offset, feature_dimension_);
  w_minus_ += x.segment(feature_dimension_, feature_dimension_) -
    x.segment(variable_offset + feature_dimension_, feature_dimension_);
  int count = offset_v, index = 0;
  for (int i = 0; i < num_positive_bags_; ++i) {
    int num_instance = positive_instances_per_bag_(i);
    v_.segment(index + 1, num_instance - 1) +=
      x.segment(count, num_instance - 1) -
      x.segment(variable_offset + count, num_instance - 1);
    v_(index) = 1.0 - v_.segment(index + 1, num_instance - 1).lpNorm<1>();
    count += num_instance - 1;
    index += num_instance;
  }
  bias_ += x(offset_bias) - x(variable_offset + offset_bias);
  error_positive_ += x.segment(offset_error_positive, num_positive_bags_) -
    x.segment(variable_offset + offset_error_positive, num_positive_bags_);
  error_negative_ += x.segment(offset_error_negative, num_negative_instances_) -
    x.segment(variable_offset + offset_error_negative, num_negative_instances_);
  w_ = w_plus_ - w_minus_;
}

void WibergMultipleInstanceLearning::
ExtractWibergSolutionfromLinearProgram(const Eigen::VectorXd &x) {
  int offset_bias = num_positive_instances_ - num_positive_bags_;
  int offset_error_positive = offset_bias + 1;
  int offset_error_negative = offset_error_positive + num_positive_bags_;
  int variable_offset = offset_error_negative + num_negative_instances_;
  // x is of the form [dv db derr+ derr-]+ and [dv db derr+ derr-]-.
  int count = 0, index = 0;
  for (int i = 0; i < num_positive_bags_; i++) {
    int num_instance = positive_instances_per_bag_(i);
    v_.segment(index + 1, num_instance - 1) +=
      x.segment(count, num_instance - 1) -
      x.segment(variable_offset + count, num_instance - 1);
    v_(index) = 1.0 - v_.segment(index + 1, num_instance - 1).lpNorm<1>();
    count += num_instance - 1;
    index += num_instance;
  }
  bias_ += x(offset_bias) - x(variable_offset + offset_bias);
  error_positive_ += x.segment(offset_error_positive, num_positive_bags_) -
    x.segment(variable_offset + offset_error_positive, num_positive_bags_);
  error_negative_ += x.segment(offset_error_negative, num_negative_instances_) -
    x.segment(variable_offset + offset_error_negative, num_negative_instances_);
  // dw is the approximation of dw/dv * dv.
  VectorXd dw = derivative_w_wrt_v_ * (x.segment(0, num_positive_instances_ -
    num_positive_bags_) - x.segment(variable_offset, num_positive_instances_ -
    num_positive_bags_));
  w_plus_ += dw.segment(0, feature_dimension_);
  w_minus_ += dw.segment(feature_dimension_, feature_dimension_);
  w_ = w_plus_ - w_minus_;
}

// Calculate the average instances in positive bags.
MatrixXd WibergMultipleInstanceLearning::CalculateAveragePositiveInstance() {
  MatrixXd average_positive_instance;
  average_positive_instance = MatrixXd::Zero(
      num_positive_bags_, feature_dimension_);
  int count = 0;
  for (int i = 0; i < num_positive_bags_; ++i) {
    for (int j = 0; j < positive_instances_per_bag_(i); ++j) {
      average_positive_instance.row(i) += v_(count) * positive_bags_.row(count);
      ++count;
    }
  }
  return average_positive_instance;
}

// Calculate the derivatives of w w.r.t. v.
void WibergMultipleInstanceLearning::GetDerivative_W_Wrt_V() {
  // Get basis solution.
  basis_.Compute(linear_program_);

  // Compute dw/dA+ since only A+ is related to v.
  // The linear program solution x is [w+ w- err+ err- b+ b-]^T, and A are
  // constraints on err+ and err- (i.e. [A+ A-]), only A+ is related with v.
  MatrixXd derivative_w_wrt_A_plus;
  derivative_w_wrt_A_plus.resize(2 * feature_dimension_, num_positive_bags_ *
    2 * feature_dimension_);
  for (int i = 0; i < 2 * feature_dimension_; i++) {
    for (int j = 0; j < num_positive_bags_; j++) {
      for (int k = 0; k < 2 * feature_dimension_; k++) {
        int x_B_component = basis_.A_column_to_basis_column_map()[i],
            B_column = basis_.A_column_to_basis_column_map()[k];
        if (x_B_component != -1 && B_column != -1) {
          derivative_w_wrt_A_plus(i, k * num_positive_bags_ + j) =
            -basis_.x_B()(B_column) * basis_.B_inverse()(x_B_component, j);
        } else {
          derivative_w_wrt_A_plus(i, k * num_positive_bags_ + j) = 0;
        }
      }
    }
  }

  // Compute dA+/dv.
  MatrixXd derivative_A_plus_wrt_v = MatrixXd::Zero(num_positive_bags_ *
    2 * feature_dimension_, num_positive_instances_ - num_positive_bags_);
  int count = 0, index = 0;
  for (int i = 0; i < num_positive_bags_; i++) {
    for (int j = 0; j < 2 * feature_dimension_; j++) {
      for (int k = 0; k < positive_instances_per_bag_(i) - 1; k++) {
        if (j < feature_dimension_) {
          derivative_A_plus_wrt_v(j * num_positive_bags_ + i, count + k) =
            positive_bags_(index, j) - positive_bags_(index + k + 1, j);
        } else {
          derivative_A_plus_wrt_v(j * num_positive_bags_ + i, count + k) =
            positive_bags_(index + k + 1, j - feature_dimension_) -
              positive_bags_(index, j - feature_dimension_);
        }
      }
    }
    count += positive_instances_per_bag_(i) - 1;
    index += positive_instances_per_bag_(i);
  }

  // Calculate dw/dv = dw/dA+ * dA+/dv.
  derivative_w_wrt_v_ = derivative_w_wrt_A_plus * derivative_A_plus_wrt_v;
}

// Test the derivatives of w w.r.t. v.
void WibergMultipleInstanceLearning::TestDerivative() {
  // Model parameters from previous iteration.
  VectorXd old_w, old_v, old_w_plus, old_w_minus, temp;
  temp.resize(2 * feature_dimension_);
  // Test Derivative
  SolveW();
  old_w = w_;
  old_v = v_;
  old_w_plus = w_plus_;
  old_w_minus = w_minus_;

  MatrixXd derivative_w_wrt_v = MatrixXd::Zero(2 * feature_dimension_,
    num_positive_instances_ - num_positive_bags_);
  static const double kDelta = 0.0000001;
  int index = 0;
  int num_col = 0;
  for (int i = 0; i < num_positive_bags_; i++) {
    int num_instance = positive_instances_per_bag_(i);
    for (int j = 0; j < num_instance - 1; j++) {
      v_(index) -= kDelta;
      v_(index + j + 1) += kDelta;
      SolveW();
      temp.segment(0, feature_dimension_) = (w_plus_ - old_w_plus) / kDelta;
      temp.segment(feature_dimension_, feature_dimension_) =
        (w_minus_ - old_w_minus) / kDelta;
      derivative_w_wrt_v.col(num_col) = temp;
      v_ = old_v;
      w_plus_ = old_w_plus;
      w_minus_ = old_w_minus;
      w_ = old_w;
      num_col++;
    }
    index += num_instance;
  }
}

// Compute the objective on current solution.
void WibergMultipleInstanceLearning::ComputeObjective() {
  objective_ = w_.lpNorm<1>() + lambda1_ * error_positive_.lpNorm<1>() +
    lambda2_ * error_negative_.lpNorm<1>();
}

// Project current solution to a feasible solution by Naive projection.
void WibergMultipleInstanceLearning::ComputeFeasibleObjective() {
  // Calculate average positive instance for each positive bag.
  MatrixXd average_positive_instance = CalculateAveragePositiveInstance();

  // Calculate the predictions.
  VectorXd positive_bag_prediction, negative_instance_prediction;
  positive_bag_prediction = average_positive_instance * w_ +
    VectorXd::Constant(num_positive_bags_, 1, bias_);
  negative_instance_prediction = negative_bags_ * w_ +
    VectorXd::Constant(num_negative_instances_, 1, bias_);

  // Calculate the true objective.
  true_objective_ = objective_;
  for (int i = 0; i < num_positive_bags_; i++) {
    if (positive_bag_prediction(i) < 1.0 - error_positive_(i)) {
      true_objective_ += 1.0 - positive_bag_prediction(i) - error_positive_(i);
    }
  }
  for (int i = 0; i < num_negative_instances_; i++) {
    if (-negative_instance_prediction(i) < 1.0 - error_negative_(i)) {
      true_objective_ += 1.0 + negative_instance_prediction(i) -
        error_negative_(i);
    }
  }
}

// Decide whether current solution is feasible.
bool WibergMultipleInstanceLearning::DecideFeasible() {
  // Calculate average positive instance for each positive bag.
  MatrixXd average_positive_instance = CalculateAveragePositiveInstance();

  // Calculate the prediction.
  VectorXd positive_bag_prediction, negative_instance_prediction;
  positive_bag_prediction = average_positive_instance * w_ +
    VectorXd::Constant(num_positive_bags_, 1, bias_);
  negative_instance_prediction = negative_bags_ * w_ +
    VectorXd::Constant(num_negative_instances_, 1, bias_);

  static const double kEpsilon = 0.0000001;
  // Check positive bag constraints: w'B'v + bias >= 1 - error_positive.
  for (int i = 0; i < num_positive_bags_; i++) {
    if (positive_bag_prediction(i) + kEpsilon < 1.0 - error_positive_(i))
      return false;
  }
  // Check negative instance constraints: -w'B' - bias >= 1 - error_negative.
  for (int i = 0; i < num_negative_instances_; i++) {
    if (-negative_instance_prediction(i) + kEpsilon <
      1.0 - error_negative_(i)) {
      return false;
    }
  }
  // Check w+ >= 0 and w- >= 0.
  for (int i = 0; i < feature_dimension_; i++) {
    if (w_plus_(i) + kEpsilon < 0.0)
      return false;
    if (w_minus_(i) + kEpsilon < 0.0)
      return false;
  }
  return true;
}

}  // namespace wiberg
