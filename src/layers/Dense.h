//
// Created by Boris Osaulenko on 05/07/2026.
//

#pragma once

#include "../activation/ActivationFunction.h"
#include "../lr_optimizer/Adam.h"
#include "../matrix/AlgebraicTuple.h"
#include "../matrix/Matrix.h"
#include "NoParamsLayer.h"
#include "ParamsLayer.h"

using value_type = double;

template <int i, int o>
using GradientType = AlgebraicTuple<Matrix<o, i>, Vector<o>>;

template <int i, int o, std::size_t batch_size>
class DenseLayer final
    : public ParamsLayer<Vector<i>, Vector<o>, GradientType<i, o>, batch_size> {
public:
  using Input = Vector<i>;
  using Output = Vector<o>;

private:
  using json = nlohmann::json;
  using Weights = Matrix<o, i>;
  using Bias = Vector<o>;
  using Gradient = GradientType<i, o>;
  using BackwardResult =
      ParamsLayer<Vector<i>, Vector<o>, Gradient, batch_size>::BackwardResult;

public:
  explicit DenseLayer(ActivationFunctionEnum activation_function_type,
                      const AdamFactory &optimizer_factory)
      : ParamsLayer<Vector<i>, Vector<o>, Gradient, batch_size>(
            activation_function_type, optimizer_factory) {}

  explicit DenseLayer(Gradient params,
                      ActivationFunctionEnum activation_function_type,
                      const AdamFactory &optimizer_factory)
      : ParamsLayer<Vector<i>, Vector<o>, Gradient, batch_size>(
            params, activation_function_type, optimizer_factory) {}

  Weights get_weights() const { return this->params.template get<0>(); }
  Bias get_bias() const { return this->params.template get<1>(); }

protected:
  Output compute_z(const Input &inp) const override {
    return get_weights() * inp + get_bias();
  }

  BackwardResult compute_delta(const Output &dz,
                               const Input &a_previous) const override {
    Bias delta_bias = dz;
    Weights delta_weights = Weights::outer_product(dz, a_previous);
    Input new_delta = get_weights().transpose() * dz;

    return BackwardResult{new_delta, Gradient{delta_weights, delta_bias}};
  }

private:
  std::tuple<Input, Weights, Bias>
  backward_helper(const Output &delta, const Input &a_previous_cache,
                  const Output &z_cache) const { // delta = dL/da^L
    Output dz = delta.hadamard(this->activation_function->derivative(z_cache));
    Bias delta_bias = dz;
    Weights delta_weights = Weights::outer_product(dz, a_previous_cache);
    Input new_delta = get_weights().transpose() * dz;

    return std::make_tuple(new_delta, delta_weights, delta_bias);
  }
};