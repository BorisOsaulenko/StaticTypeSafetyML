//
// Created by Boris Osaulenko on 05/07/2026.
//

#pragma once

#include "../activation/ActivationFunction.h"
#include "../lr_optimizer/Adam.h"
#include "../matrix/AlgebraicTuple.h"
#include "../matrix/Matrix.h"
#include "LayerConcepts.h"
#include "ParamsLayer.h"
#include <cstddef>

template <size_t i, size_t o>
using GradientType = AlgebraicTuple<Matrix<o, i>, Vector<o>>;

#define INPUT Vector<i>
#define OUTPUT Vector<o>
#define CACHE Cache<i, o>

template <size_t i, size_t o>
class DenseLayer final
    : public ParamsLayer<DenseLayer<i, o>, GradientType<i, o>> {
private:
  using Weights = Matrix<o, i>;
  using Bias = Vector<o>;
  using Grad = GradientType<i, o>;

  template <size_t _i, size_t _o> struct Cache {
    Vector<_i> a_previous;
    Vector<_o> z_cache;
  };

public:
  explicit DenseLayer(ActivationFunctionEnum activation_function_type,
                      const AdamFactory &optimizer_factory)
      : ParamsLayer<DenseLayer, Grad>(activation_function_type,
                                      optimizer_factory) {}

  explicit DenseLayer(Grad params,
                      ActivationFunctionEnum activation_function_type,
                      const AdamFactory &optimizer_factory)
      : ParamsLayer<DenseLayer, Grad>(params, activation_function_type,
                                      optimizer_factory) {}

  Weights get_weights() const { return this->params.template get<0>(); }
  Bias get_bias() const { return this->params.template get<1>(); }

  Weights get_weights_gradients() const {
    return this->gradient.template get<0>();
  }
  Bias get_bias_gradients() const { return this->gradient.template get<1>(); }

  OUTPUT compute_z(const INPUT &inp) const {
    return get_weights() * inp + get_bias();
  }

  ForwardCachedResult<OUTPUT, CACHE> compute_z_cached(const INPUT &inp) const {
    OUTPUT z = compute_z(inp);
    return {z, Cache{inp, z}};
  }

  BackwardResult<INPUT, Grad> compute_delta(const OUTPUT &delta,
                                            const CACHE &cache) const {
    OUTPUT dz =
        delta.hadamard(this->activation_function->derivative(cache.z_cache));
    Bias delta_bias = dz;
    Weights delta_weights = Weights::outer_product(dz, cache.a_previous);
    INPUT new_delta = get_weights().transpose() * dz;

    return {new_delta, Grad{delta_weights, delta_bias}};
  }
};

#undef INPUT
#undef OUTPUT
#undef CACHE