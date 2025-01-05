// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <numeric>
#include<math.h>
#include<iostream>
using namespace Rcpp;
using namespace std;

// [[Rcpp::export]]
double meanC(NumericVector x) {
  int n = x.size();
  double total = 0;

  for(int i = 0; i < n; ++i) {
    total += x[i];
  }
  return total / n;
}

//iterated product of vector elements
// [[Rcpp::export]]
double prodC(NumericVector x){
  double total=1;
  NumericVector::iterator it;
  for(it = x.begin(); it != x.end(); ++it) {
    total *= *it;
  }
  return total;
}

//iterated product using C++11 auto (range-based for loops)
// [[Rcpp::export]]
double prodC11(NumericVector xp){
  double total=1;
  for(const auto &x:xp) {
    total *= x;
  }
  return total;
}

// [[Rcpp::export]]
double innerprod(NumericVector& x, NumericVector& y){
  double sum=0;
  for(int i=0; i<x.size();i++){
    sum +=x[i]*y[i];
  }
  return sum;
}

/*Hermites look-up table */
const double H0[] ={
    5.00000000e-01, 4.16666667e-02, 1.25000000e-02, 5.58035714e-03,
       3.03819444e-03, 1.86434659e-03, 1.23948317e-03, 8.72802734e-04,
       6.41766716e-04, 4.88080476e-04, 3.81378900e-04, 3.04688578e-04,
       2.47969627e-04, 2.05001345e-04, 1.71776989e-04, 1.45629484e-04,
       1.24732562e-04, 1.07804571e-04, 9.39264577e-05, 8.24264876e-05,
       7.28052774e-05, 6.46858728e-05, 5.77797965e-05, 5.18635141e-05,
       4.67618378e-05, 4.23360380e-05, 3.84752043e-05, 3.50898860e-05,
       3.21073518e-05, 2.94680187e-05, 2.71227322e-05, 2.50306762e-05,
       2.31577515e-05, 2.14753073e-05, 1.99591417e-05, 1.85887077e-05,
       1.73464782e-05, 1.62174355e-05, 1.51886571e-05, 1.42489791e-05,
       1.33887201e-05, 1.25994539e-05, 1.18738219e-05, 1.12053766e-05,
       1.05884513e-05, 1.00180510e-05, 9.48975982e-06, 8.99966367e-06,
       8.54428410e-06, 8.12052213e-06
};

/**
 // [[Rcpp::export]]
 double KHC11(double a, double b,int l=3) {
 const double  Cw=2.5, Cb=1.0;
 const int to=30;
 NumericVector vro(to),vherm(to);
 if (l == 0) { cout<<"l==0 :"<<a*b*Cw + Cb<<endl;
 return a*b*Cw + Cb;  }  //std::inner_product(a.begin(), a.end(), b.begin(), 0)
 else { double ro = KHC11(a,b,l-1) / sqrt(KHC11(a,a,l-1)*KHC11(b,b,l-1));
 cout<<ro<<endl;
 for(int k=0;k<to;k++){vro[k]=pow(ro,2*k+3); cout<<"vro"<<k<<":"<<vro[k]<<endl;}
 for(int k=0;k<to;k++){vherm[k]=H0[k]; cout<<"vherm"<<k<<":"<<vherm[k]<<endl;}
 cout<<1/4.0 + 1/(2*M_PI)*(ro + innerprod(vherm, vro))<<endl;
 return 1/2.0 + 1/(2*M_PI)*(ro + innerprod(vherm, vro));
 // rm(vro,vherm)
 }
 }
 **/

// [[Rcpp::export]]
double KHC11(NumericVector& a, NumericVector& b,int l=3) {
  const double  Cw=2.5, Cb=1.0;
  const int to=50;  //30, 50 tarda eternidad
  double sigma1, sigma2;
  int i;
  NumericVector vro(to),vherm(to);
  if (l == 0) {// cout<<"l==0 :"<<innerprod(a,b)*Cw/a.size() + Cb<<endl;
    return innerprod(a,b)*Cw/a.size() + Cb;  }  //std::inner_product(a.begin(), a.end(), b.begin(), 0)
  else { double ro = KHC11(a,b,l-1)/sqrt(KHC11(a,a,l-1)*KHC11(b,b,l-1));
    sigma1 = pow(KHC11(a,a,l-1),1/2.);
    sigma2 = pow(KHC11(b,b,l-1),1/2.);
   // cout<<"ro= "<<ro<<endl;
    for(int i=0;i<to;i++){vro[i]=pow(ro,2*i+2);}
    for(int i=0;i<to;i++){vherm[i]=H0[i];}
  // cout<<"l= "<<l-1<<" : "<<1/4.0 + 1/(2*M_PI)*(ro + innerprod(vherm, vro))<<endl;
    return sigma1*sigma2*(1/(2.0*M_PI) + 1/4*ro + 1/(2*M_PI)*innerprod(vherm, vro));
    // rm(vro,vherm)
  }
}


