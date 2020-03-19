#include "fast_mass_springs_precomputation_sparse.h"
#include "signed_incidence_matrix_sparse.h"
#include <vector>

bool fast_mass_springs_precomputation_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::SparseMatrix<double>  & M,
  Eigen::SparseMatrix<double>  & A,
  Eigen::SparseMatrix<double>  & C,
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization)
{
  /////////////////////////////////////////////////////////////////////////////
  // Replace with your code
  std::vector<Eigen::Triplet<double> > ijv;
  const int n = V.rows();
  // for(int i = 0;i<n;i++) ijv.emplace_back(i,i,1);
  Eigen::SparseMatrix<double> Q(n,n);
  // Q.setFromTriplets(ijv.begin(),ijv.end());

  r = Eigen::VectorXd::Zero(E.rows());
  for(int i=0; i<E.rows();i++) r(i) = (V.row(E(i,0)) - V.row(E(i,1))).norm();

  std::vector<Eigen::Triplet<double>> ijv_m;
  M.resize(n,n);
  for(int i=0; i<V.rows();i++) ijv_m.emplace_back(i,i,m(i));
  M.setFromTriplets(ijv_m.begin(), ijv_m.end());

  signed_incidence_matrix_sparse(V.rows(), E, A);

  std::vector<Eigen::Triplet<double>> ijv_c;
  C.resize(b.size(),n);
  for(int i=0; i<b.size();i++) ijv_c.emplace_back(i,b[i],1);
  C.setFromTriplets(ijv_c.begin(), ijv_c.end());

  double w = 1e10;

  Q = k * A.transpose() * A + M/(delta_t * delta_t);
  Q += w * C.transpose() * C;
  
  /////////////////////////////////////////////////////////////////////////////
  prefactorization.compute(Q);
  return prefactorization.info() != Eigen::NumericalIssue;
}
