//
// Created by Boris Osaulenko on 05/07/2026.
//

#pragma once

#include "../activation/ActivationFunction.h"
#include "../lr_optimizer/Adam.h"
#include "../matrix/AlgebraicTuple.h"
#include "../matrix/Matrix.h"
#include "ParamsLayer.h"

using value_type = double;

template <int i, int o>
using GradientType = AlgebraicTuple<Matrix<o, i>, Vector<o>>;

template <int i, int o>
class DenseLayer final
    : public ParamsLayer<DenseLayer<i, o>, GradientType<i, o>> {
public:
  using Input = Vector<i>;
  using Output = Vector<o>;

private:
  using json = nlohmann::json;
  using Weights = Matrix<o, i>;
  using Bias = Vector<o>;
  using Gradient = GradientType<i, o>;
  using BackwardResult = LayerBackwardResult<Input, Gradient>;

public:
  explicit DenseLayer(ActivationFunctionEnum activation_function_type,
                      const AdamFactory &optimizer_factory)
      : ParamsLayer<DenseLayer, Gradient>(activation_function_type,
                                          optimizer_factory) {}

  explicit DenseLayer(Gradient params,
                      ActivationFunctionEnum activation_function_type,
                      const AdamFactory &optimizer_factory)
      : ParamsLayer<DenseLayer, Gradient>(params, activation_function_type,
                                          optimizer_factory) {}

  Weights get_weights() const { return this->params.template get<0>(); }
  Bias get_bias() const { return this->params.template get<1>(); }

  Weights get_weights_gradients() const {
    return this->gradient.template get<0>();
  }
  Bias get_bias_gradients() const { return this->gradient.template get<1>(); }

  Output compute_z(const Input &inp) const {
    return get_weights() * inp + get_bias();
  }

  BackwardResult compute_delta(const Output &delta, const Input &a_previous,
                               const Output &z_cache) const {
    Output dz = delta.hadamard(this->activation_function->derivative(z_cache));
    Bias delta_bias = dz;
    Weights delta_weights = Weights::outer_product(dz, a_previous);
    Input new_delta = get_weights().transpose() * dz;

    return BackwardResult{new_delta, Gradient{delta_weights, delta_bias}};
  }
};
