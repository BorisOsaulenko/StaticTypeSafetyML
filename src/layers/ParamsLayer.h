//
// Created by Boris Osaulenko on 19/07/2026.
//

#ifndef ML_PARAMSLAYER_H
#define ML_PARAMSLAYER_H
#include "../activation/ActivationFactory.h"
#include "../activation/ActivationFunction.h"
#include "../lr_optimizer/Adam.h"
#include "../matrix/Algebraic.h"

template <Algebraic Input, Algebraic Output, Algebraic Grad> class ParamsLayer {
  using json = nlohmann::json;

public:
  static constexpr bool has_params = true;
  using Params = Grad;

  explicit ParamsLayer(const ActivationFunctionEnum activation_function_type,
                       const AdamFactory &optimizer_factory)
      : activation_function(
            ActivationFactory::create(activation_function_type)),
        params(Grad::random()), gradient(Grad::zeros()),
        optimizer(optimizer_factory.build(params)) {}

  explicit ParamsLayer(Grad &params,
                       const ActivationFunctionEnum activation_function_type,
                       const AdamFactory &optimizer_factory)
      : activation_function(
            ActivationFactory::create(activation_function_type)),
        params(params), gradient(Grad::zeros()),
        optimizer(optimizer_factory.build(params)) {}

  virtual ~ParamsLayer() = default;

  struct BackwardResult {
    Input new_delta;
    Grad gradient;
  };

  template <std::size_t SIZE>
  AlgebraicArray<SIZE, Output>
  forward_vectorized(const AlgebraicSpan<SIZE, const Input> &inp,
                     AlgebraicArray<SIZE, Output> &z_cache) const {
    AlgebraicArray<SIZE, Output> result;

#pragma omp parallel for
    for (std::size_t i = 0; i < inp.size(); ++i) {
      Output z = compute_z(inp[i]);
      z_cache[i] = z;
      result[i] = activation_function->activate(z);
    }

    return result;
  }

  Output forward_no_cache(const Input &inp) const {
    Output z = compute_z(inp);
    return activation_function->activate(z);
  }

  template <std::size_t SIZE>
  AlgebraicArray<SIZE, Output> forward_no_cache_vectorized(
      const AlgebraicSpan<SIZE, const Input> &inp) const {
    AlgebraicArray<SIZE, Output> result;

#pragma omp parallel for
    for (std::size_t i = 0; i < inp.size(); ++i) {
      result[i] = activation_function->activate(compute_z(inp[i]));
    }

    return result;
  }

  template <std::size_t SIZE>
  AlgebraicArray<SIZE, Input>
  backward_vectorized(const AlgebraicSpan<SIZE, const Output> &delta,
                      const AlgebraicSpan<SIZE, const Input> &a_previous,
                      const AlgebraicSpan<SIZE, const Output> &z_cache) {
    AlgebraicArray<SIZE, Input> result;

#pragma omp parallel
    {
      Grad local_grad = Grad::zeros();

#pragma omp for
      for (std::size_t idx = 0; idx < SIZE; ++idx) {
        auto [new_delta, grad] =
            compute_delta(delta[idx], a_previous[idx], z_cache[idx]);

        result[idx] = new_delta;
        local_grad += grad;
      }

#pragma omp critical
      gradient += local_grad;
    }

    return result;
  }

  void apply_change(const std::size_t &normalize_by) {
    gradient /= static_cast<Grad::value_type>(normalize_by);
    optimizer.adjust(gradient);
    gradient = Grad::zeros();
  }

  friend void to_json(json &j, const ParamsLayer &layer) {
    j = json{{"params", layer.params},
             {"activation_function", layer.activation_function->get_type()}};
  }

  friend void from_json(const json &j, ParamsLayer &layer) {
    layer.params = j.at("params").get<Grad>();
    const auto activation_type =
        j.at("activation_function").get<ActivationFunctionEnum>();
    layer.activation_function = ActivationFactory::create(activation_type);
  }

protected:
  std::unique_ptr<ActivationFunction> activation_function;
  Grad params;
  Grad gradient;

  using value_type = double;
  virtual Output compute_z(const Input &inp) const = 0;
  virtual BackwardResult compute_delta(const Output &delta,
                                       const Input &a_previous,
                                       const Output &z_cache) const = 0;

private:
  Adam<Grad> optimizer;
};

#endif // ML_PARAMSLAYER_H
