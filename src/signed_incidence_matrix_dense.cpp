#include "signed_incidence_matrix_dense.h"

void signed_incidence_matrix_dense(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::MatrixXd & A)
{
  //////////////////////////////////////////////////////////////////////////////
  // Replace with your code
  A = Eigen::MatrixXd::Zero(E.rows(),n);
  for(int e=0; e<A.rows(); e++){
    A(e, E(e,0)) = 1;
    A(e, E(e,1)) = -1;
  }
  //////////////////////////////////////////////////////////////////////////////
}
