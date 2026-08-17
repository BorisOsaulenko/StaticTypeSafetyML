//
// Created by Boris Osaulenko on 05/07/2026.
//

#ifndef ML_LAYER_H
#define ML_LAYER_H
#include "../matrix/Algebraic.h"
#include "../matrix/StaticArray.h"
#include "LayerConcepts.h"

#include <nlohmann/json.hpp>
#include <type_traits>

template <typename Derived> class NoParamsLayer {
  using json = nlohmann::json;

  const Derived &derived() const { return static_cast<const Derived &>(*this); }

public:
  static constexpr bool has_params = false;
  explicit NoParamsLayer() = default;

  template <typename Input>
  auto forward_no_cache(const Input &inp) const
    requires requires { this->derived().compute_z(inp); }
  {
    return this->derived().compute_z(inp);
  }

  template <std::size_t SIZE, typename Input, typename Output>
  StaticArray<SIZE, Output>
  forward_vectorized(const StaticSpan<SIZE, const Input> &inp,
                     StaticArray<SIZE, Output> &z_cache) const
    requires ComputesZ<Derived, Input, Output>
  {
    StaticArray<SIZE, Output> z;

#pragma omp parallel for
    for (std::size_t i = 0; i < inp.size(); ++i) {
      z[i] = this->derived().compute_z(inp[i]);
      z_cache[i] = z[i];
    }

    return z;
  }

  template <std::size_t SIZE, typename Input>
  auto forward_no_cache_vectorized(const StaticSpan<SIZE, const Input> &inp) const
    requires requires { this->derived().compute_z(inp[0]); }
  {
    using Output = decltype(this->derived().compute_z(inp[0]));
    StaticArray<SIZE, Output> result;

#pragma omp parallel for
    for (std::size_t i = 0; i < inp.size(); ++i) {
      result[i] = this->derived().compute_z(inp[i]);
    }

    return result;
  }

  template <std::size_t SIZE, typename Input, typename Output>
  StaticArray<SIZE, Input>
  backward_vectorized(
      const StaticSpan<SIZE, const Output> &delta,
      const StaticSpan<SIZE, const Input> &a_previous,
      const StaticSpan<SIZE, const std::type_identity_t<Output>> &z_cache) const
    requires NoParamsLayerImpl<Derived, Input, Output>
  {
    StaticArray<SIZE, Input> new_delta;

#pragma omp parallel for
    for (std::size_t i = 0; i < delta.size(); ++i) {
      new_delta[i] =
          this->derived().compute_delta(delta[i], a_previous[i], z_cache[i]);
    }
    return new_delta;
  }
};

#endif // ML_LAYER_H
