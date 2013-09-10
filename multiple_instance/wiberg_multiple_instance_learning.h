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

// Three optimization methods for multiple instance learning including
// alternative, simultaneous and wiberg optimizations.
// The optimization problem of multiple instance learning is:
//
//   min |w| + lambda1 * sum (error_positive) + lambda2 * sum (error_negative)
//    s.t.   w' * B_positive' * v + bias >= 1 - error_positive
//             -w' * B_negative' - bias >= 1 - error_negative
//        sum (v) = 1, v >= 0, error_positive >=0, error_negative >= 0
//
//   where lambda1 and lambda2 are weights parameters. w and bias are linear
//   classifier. v is the combination coefficients for positive bags.
//   error_positive is the classification error of positive bags, and
//   error_negative is the classification error of negative instances.
//   B_positive and B_negative are the positive bags and negative instances.
//   The unknowns here are w, v, bias, error_positive and error_negative.
//
//   Initialization:
//   w and bias are set to be 0. v is set to be equal.
//   error_positive and error_negative are set to be 1. Note that this
//   initialization is a feasible solution to the problem (feasible means all
//   constraints are satisified).
//
//   Projection:
//   After each iteration, we have a new solution. However, this solution can
//   be infeasible in simultaneous and wiberg methods since we approximate the
//   non-linear constraints. Therefore, we do a projection to project the
//   infeasible solution back to a feasible point.
//   Two projections methods are used: The first naive projection is
//   fixing v, w and bias, modifying error_positive to satisfy the
//   constraints. The second projection is fixing v only, solve another exact
//   linear program to find a feasible solution.
//
//   Alternative Optimization:
//   We alternatively optimize over w and v by fixing the other set of
//   variables. Each problem is to solve a standard linear program.
//
//   Simultaneous Optimization:
//   We linearize the non-linear constraints and split w into w+ and w- to
//   find a new solution via solving a linear program. In the linear program,
//   the unknowns are of step sizes: [dw+ dw- dv db derr+ derr-]. Since there
//   is no constraints on the step sizes, we need to further split them into
//   [dw+ dw- dv db derr+ derr-]+ and [dw+ dw- dv db derr+ derr-]- to form a
//   linear program. We also add a trust region to control the step size.
//   We use the first order Taylor expansion to linearize the constraints.
//
//   Wiberg Optimization:
//   We only explicitly optimize over variable v only and implicitly solve w.
//   Given v, we can obtain the optimial w by solving a linear program. Thus,
//   w can be represent as a function of v, i.e., w(v) and we can calculate
//   the derivatives of w w.r.t. v, i.e., dw/dv. We then linearize the
//   objective and constraints to form a linear program of variable v only.
//   In this linear program, the unknowns are the step sizes:
//   [dv, db, derr+, derr-]. Since there are no constraints on the step size,
//   we need to further split them into [dv, db, derr+, derr-]+ and
//   [dv, db, derr+, derr-]- to form the linear program. We also add a trust
//   region to control the step size.
#ifndef WIBERG_MULTIPLE_INSTANCE_WIBERG_MULTIPLE_INSTANCE_LEARNING_H_
#define WIBERG_MULTIPLE_INSTANCE_WIBERG_MULTIPLE_INSTANCE_LEARNING_H_

#include "l1/linear_system/linear_program.h"
#include "l1/linear_system/basis.h"
#include "eigen3/Eigen/Dense"

namespace wiberg {

class WibergMultipleInstanceLearning {
 public:
  WibergMultipleInstanceLearning() {}
  ~WibergMultipleInstanceLearning() {}
  // Three different optimization methods.
  void SolveAlternative();
  void SolveSimultaneous();
  void SolveWiberg();
  // Initialize data and model parameters.
  void Init(const Eigen::MatrixXd &positive_bags,
            const Eigen::MatrixXd &negative_bags,
            const Eigen::VectorXd &positive_instances_per_bag,
            const Eigen::VectorXd &negative_instances_per_bag,
            double lambda1, double lambda2);
  // Get data and model parameters.
  const Eigen::MatrixXd &positive_bags() const { return positive_bags_; }
  const Eigen::MatrixXd &negative_bags() const { return negative_bags_; }
  const Eigen::VectorXd &positive_instances_per_bag() const
    { return positive_instances_per_bag_; }
  const Eigen::VectorXd &negative_instances_per_bag() const
    { return negative_instances_per_bag_; }
  const Eigen::VectorXd &w() const { return w_; }
  const Eigen::VectorXd &v() const { return v_; }
  const double &bias() const { return bias_; }

 private:
  // Four linear programs.
  void SolveW();
  void SolveV();
  void SolveSimultaneousLinearProgram();
  void SolveWibergLinearProgram();
  // Extract solutions from different linear programs.
  void ExtractWfromLinearProgram(const Eigen::VectorXd &x);
  void ExtractVfromLinearProgram(const Eigen::VectorXd &x);
  void ExtractSimultaneousSolutionfromLinearProgram(const Eigen::VectorXd &x);
  void ExtractWibergSolutionfromLinearProgram(const Eigen::VectorXd &x);
  // Calculate the average instances in positive bags.
  Eigen::MatrixXd CalculateAveragePositiveInstance();
  // Calculate the derivative of w with respect to v.
  void GetDerivative_W_Wrt_V();
  // Test the derivatives.
  void TestDerivative();
  // Compute the objective on current solution.
  void ComputeObjective();
  // Project current solution to a feasible solution by Naive projection.
  void ComputeFeasibleObjective();
  // Decide whether the solution is feasible.
  bool DecideFeasible();

  // These are problem data.
  LinearProgram linear_program_;
  Basis basis_;
  Eigen::MatrixXd positive_bags_, negative_bags_, derivative_w_wrt_v_;
  Eigen::VectorXd positive_instances_per_bag_, negative_instances_per_bag_,
    w_, v_, w_plus_, w_minus_, error_positive_, error_negative_;
  int num_positive_instances_, num_negative_instances_, feature_dimension_,
    num_positive_bags_, num_negative_bags_, max_iterations_;
  double lambda1_, lambda2_, bias_, objective_, true_objective_, mu_, epsilon_;
};

}  // namespace wiberg

#endif  // WIBERG_MULTIPLE_INSTANCE_WIBERG_MULTIPLE_INSTANCE_LEARNING_H_
