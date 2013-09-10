WHAT IS IT

This code implements Wiberg minimization for L1 matrix factorization
and multiple instance learning.

Wiberg minimization is an algorithm for chicken-and-egg minimization problems
with better convergence than the standard chicken-and-egg algorithm,
expectation-maximization (EM).  To minimize a function f(U, V) with respect to
two sets of variables U and V, Wiberg solves for V given U, linearizes the
solution V(U) with respect to U, then minimizes f with respect to U
only, letting V vary implicitly via the linearization.

For L1 matrix factorization and multiple instance learning, the Wiberg
minimization with respect to U is done using successive linear programming.
It's also possible to instead minimize with respect to both U and V
simultaneously using successive linear programming.  This approach
is competitive with Wiberg and like Wiberg, converges better than EM.
So, we've also included implementations for matrix factorization and
multiple instance learning via successive linear programming.


MORE INFORMATION

These papers are the most closely related to the code here.  More general
information on Wiberg minimization can be found in the references in
(Eriksson and van den Hengel 2010) and (Strelow 2012).

(Eriksson and van den Hengel 2010) Anders Eriksson and Anton van den Hengel,
Efﬁcient computation of robust low-rank matrix approximations in the presence
of missing data using the L1 norm, Computer Vision and Pattern Recognition,
San Francisco, 2010.  http://cs.adelaide.edu.au/~anders/papers/
eriksson-cvpr-10.pdf.

  The original paper on Wiberg L1 matrix factorization.  Note that MATLAB code
  for this paper is also available, at http://cs.adelaide.edu.au/~anders/.

(Strelow 2012) Dennis Strelow, General and nested Wiberg minimization, Computer
Vision and Pattern Recognition, Providence, Rhode Island, 2012.
http://research.google.com/pubs/pub37749.html.

  Gives a simplified version of Wiberg L1 matrix factorization, introduces 
  successive linear programming for L1 matrix factorization, and Wiberg and
  successive linear programming algorithms for minimizing more general 
  functions f(U, V).  The code in our l1/linear_system and l1/factorization
  directories follows the development in this paper.

(Mangasarian and Wild 2008) O.L. Mangasarian and E.W. Wild, Multiple instance
classification via successive linear programming, J. Optim Theory appl (2008)
137: 555-568.

  Our Wiberg and successive linear programming MIL algorithms minimize an
  objective similar to Mangasarian and Wild's and the multiple_instance 
  directory includes a baseline EM algorithm similar to Mangasarian and Wild's.


AUTHORS

L1 matrix factorization: Dennis Strelow (strelow@google.com,
strelow@gmail.com).

Multiple instance learning: Qifan Wang (wqfcr618@gmail.com).


THANK YOU

Many thanks to Jean-Yves Bouguet, Vivek Verma, Fabien Viger, and Frédéric
Didier for code reviews and suggestions.  Thanks also to Jean-Yves Bouguet,
Anders Eriksson, and Luo Si for many helpful discussions.


DISCLAIMER

This code is not a Google product.
