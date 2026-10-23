#ifndef ML_PARAMSLAYER_H
#define ML_PARAMSLAYER_H
#include "../activation/ActivationFactory.h"
#include "../activation/ActivationFunction.h"
#include "../lr_optimizer/Optimizer.h"
#include "LayerConcepts.h"
#include <concepts>
#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>

#define OUTPUT LayerOutput<Derived, Input>
#define CACHE LayerCache<Derived, Input>

#define CACHE_ERROR                                                            \
  std::string("Cache has invalid type for ") + LayerToString<Derived> +        \
      ". Should be StaticArray<SIZE, LayerCache<Derived, Input> &"

#define DELTA_ERROR                                                            \
  std::string("Delta has invalid type for ") + LayerToString<Derived> +        \
      ". Should be StaticSpan<SIZE, LayerOutput<Derived, Input>> &"

template <typename Derived, Algebraic Grad> class ParamsLayer {
  using json = nlohmann::json;
  const Derived &derived() const { return static_cast<const Derived &>(*this); }

public:
  static constexpr bool has_params = true;
  using Params = Grad;

  explicit ParamsLayer(const ActivationFunctionEnum activation_function_type)
      : activation_function_type(activation_function_type),
        activation_function(
            ActivationFactory::create(activation_function_type)),
        params(Grad::random()), gradient(Grad::zeros()) {}

  explicit ParamsLayer(Grad &params,
                       const ActivationFunctionEnum activation_function_type)
      : activation_function_type(activation_function_type),
        activation_function(
            ActivationFactory::create(activation_function_type)),
        params(params), gradient(Grad::zeros()) {}

  explicit ParamsLayer(const ParamsLayer &other) {
    params = other.params;
    activation_function_type = other.activation_function_type;
    activation_function =
        ActivationFactory::create(other.activation_function_type);
  }

  void
  set_optimizer(const OptimizerFactoryFor<Params> auto &optimizer_factory) {
    this->optimizer = optimizer_factory.build(params);
  }

  template <typename Input>
  auto forward_no_cache(const Input &inp) const
    requires ComputesZ<Derived, Input>
  {
    return activation_function->activate(derived().compute_z(inp));
  }

  template <size_t SIZE, typename Input>
  auto forward_vectorized(const StaticArray<SIZE, Input> &inp,
                          auto &cache) const
    requires ComputesZ<Derived, Input>
  {
    return forward_vectorized(StaticSpan<SIZE, const Input>(inp), cache);
  }

  template <size_t SIZE, typename Input>
  auto forward_vectorized(const StaticSpan<SIZE, const Input> &inp,
                          auto &cache) const
    requires ComputesZ<Derived, Input>
  {
    static_assert(std::same_as<StaticArray<SIZE, LayerCache<Derived, Input>> &,
                               decltype(cache)>,
                  CACHE_ERROR);
    StaticArray<SIZE, OUTPUT> result;
#pragma omp parallel for
    for (size_t i = 0; i < inp.size(); ++i) {
      LayerForwardCachedResult<Derived, Input> r =
          derived().compute_z_cached(inp[i]);

      result[i] = activation_function->activate(r.output);
      cache[i] = r.cache;
    }

    return result;
  }

  template <size_t SIZE, typename Input>
  auto
  forward_no_cache_vectorized(const StaticSpan<SIZE, const Input> &inp) const
    requires ComputesZ<Derived, Input>
  {
    StaticArray<SIZE, OUTPUT> result;

#pragma omp parallel for
    for (size_t i = 0; i < inp.size(); ++i) {
      result[i] = activation_function->activate(derived().compute_z(inp[i]));
    }

    return result;
  }

  template <size_t SIZE, typename Input>
  StaticArray<SIZE, Input> backward_vectorized(const auto &delta,
                                               const auto &cache)
    requires ComputesDeltaCached<Derived, Input, Grad>
  {
    static_assert(
        std::same_as<const StaticArray<SIZE, OUTPUT> &, decltype(delta)>,
        DELTA_ERROR);
    static_assert(
        std::same_as<const StaticArray<SIZE, CACHE> &, decltype(cache)>,
        CACHE_ERROR);

    StaticArray<SIZE, Input> result;

#pragma omp parallel
    {
      Grad local_grad = Grad::zeros();

#pragma omp for
      for (size_t idx = 0; idx < SIZE; ++idx) {
        auto [new_delta, grad] =
            derived().compute_delta(delta[idx], cache[idx]);

        result[idx] = new_delta;
        local_grad += grad;
      }

#pragma omp critical
      gradient += local_grad;
    }

    return result;
  }

  void apply_change(const size_t &normalize_by) {
    gradient /= static_cast<Grad::value_type>(normalize_by);
    optimizer->adjust(gradient);
    gradient = Grad::zeros();
  }

  friend void to_json(json &j, const ParamsLayer &layer) {
    j = json{{"params", layer.params},
             {"activation_function", layer.activation_function->get_type()},
             {"optimizer", *layer.optimizer}};
  }

  friend void from_json(const json &j, ParamsLayer &layer) {
    layer.params = j.at("params").get<Grad>();
    const auto activation_type =
        j.at("activation_function").get<ActivationFunctionEnum>();
    layer.activation_function = ActivationFactory::create(activation_type);
    j.at("optimizer").get_to(*layer.optimizer);
  }

protected:
  ActivationFunctionEnum activation_function_type;
  std::unique_ptr<ActivationFunction> activation_function;
  Grad params;
  Grad gradient;

  using value_type = double;

private:
  std::unique_ptr<Optimizer<Grad>> optimizer;
};

#undef OUTPUT
#undef CACHE
#undef CACHE_ERROR
#undef DELTA_ERROR

#endif // ML_PARAMSLAYER_H
