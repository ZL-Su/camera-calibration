#pragma once
#include <ceres\jet.h>
#include <Eigen\Core>

namespace ceres {

// A jet traits class to make it easier to work with mixed auto / numeric diff.
template<typename T> struct JetOps
{
	using value_t = T;
	using const_ref = const value_t&;
	using pointer = value_t * ;
	static constexpr bool IsScalar() { return true; }
	static constexpr value_t GetScalar(const_ref t) { return t; }
	static constexpr void SetScalar(const_ref scalar, pointer t) { *t = scalar; }
	static constexpr void ScaleDerivative(double /*scale_by*/, pointer /*value*/) {
		// For double, there is no derivative to scale.
	}
};

template<typename T, int N> struct JetOps<Jet<T, N>> 
{
	using value_t = T;
	using const_ref = const value_t&;
	using pointer = value_t * ;
	using jet_t = Jet<value_t, N>;
   static constexpr bool IsScalar() { return false; }
   static constexpr value_t GetScalar(const jet_t& t) { return t.a; }
   static constexpr void SetScalar(const_ref scalar, jet_t* t) { t->a = scalar; }
   static constexpr void ScaleDerivative(double scale_by, jet_t* value) { value->v *= scale_by; }
};

template<typename _Fnt, int kNumArgs, typename _Arg>
struct Chain 
{
  static _Arg Rule(const _Fnt &f, const _Fnt /*dfdx*/[kNumArgs], 
	  const _Arg /*x*/[kNumArgs]) {
    // In the default case of scalars, there's nothing to do since there are no
    // derivatives to propagate.
    return f;
  }
};

// XXX Add documentation here!
template<typename _Fnt, int kNumArgs, typename T, int N>
struct Chain<_Fnt, kNumArgs, Jet<T, N> > 
{
  using value_t = T;
  using jet_t = Jet<value_t, N>;
  static jet_t Rule(const _Fnt &f, const _Fnt dfdx[kNumArgs], const jet_t x[kNumArgs])
  {
    // x is itself a function of another variable ("z"); what this function
    // needs to return is "f", but with the derivative with respect to z
    // attached to the jet. So combine the derivative part of x's jets to form
    // a Jacobian matrix between x and z (i.e. dx/dz).
    Eigen::Matrix<value_t, kNumArgs, N> dxdz;
    for (int i = 0; i < kNumArgs; ++i) {
      dxdz.row(i) = x[i].v.transpose();
    }

    // Map the input gradient dfdx into an Eigen row vector.
    Eigen::Map<const Eigen::Matrix<_Fnt, 1, kNumArgs> >
        vector_dfdx(dfdx, 1, kNumArgs);

    // Now apply the chain rule to obtain df/dz. Combine the derivative with
    // the scalar part to obtain f with full derivative information.
	 jet_t jet_f;
    jet_f.a = f;
    jet_f.v = vector_dfdx.template cast<value_t>() * dxdz;  // Also known as dfdz.
    return jet_f;
  }
};

}  // namespace ceres