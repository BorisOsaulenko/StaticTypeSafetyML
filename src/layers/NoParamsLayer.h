//
// Created by Boris Osaulenko on 05/07/2026.
//

#ifndef ML_LAYER_H
#define ML_LAYER_H
#include "../matrix/Algebraic.h"
#include "../matrix/StaticArray.h"
#include "LayerConcepts.h"

#include <nlohmann/json.hpp>

template <typename Derived> class NoParamsLayer {
  using json = nlohmann::json;

  const Derived &derived() const { return static_cast<const Derived &>(*this); }

public:
  static constexpr bool has_params = false;
  explicit NoParamsLayer() = default;

  template <typename Input>
    requires ComputesZNotCached<Derived, Input>
  Derived::template Output<Input> forward_no_cache(const Input &inp) const {
    return this->derived().compute_z_not_cached(inp);
  }

  template <std::size_t SIZE, typename Input>
    requires ComputesZCached<Derived, Input>
  StaticArray<SIZE, typename Derived::template LayerForwardCachedResult<Input>>
  forward_vectorized(const StaticSpan<SIZE, const Input> &inp) const {
    StaticArray<SIZE,
                typename Derived::template LayerForwardCachedResult<Input>>
        z;

#pragma omp parallel for
    for (std::size_t i = 0; i < inp.size(); ++i) {
      z[i] = this->derived().compute_z_cached(inp[i]);
    }

    return z;
  }

  template <std::size_t SIZE, typename Input>
    requires ComputesZNotCached<Derived, Input>
  StaticArray<SIZE, typename Derived::template Output<Input>>
  forward_no_cache_vectorized(const StaticSpan<SIZE, const Input> &inp) const {
    StaticArray<SIZE, typename Derived::template Output<Input>> result;

#pragma omp parallel for
    for (std::size_t i = 0; i < inp.size(); ++i) {
      result[i] = this->derived().compute_z_not_cached(inp[i]);
    }

    return result;
  }

  template <std::size_t SIZE, typename Input>
    requires ComputesDeltaDirect<Derived, Input>
  StaticArray<SIZE, Input> backward_vectorized(
      const StaticSpan<SIZE, const typename Derived::template Output<Input>>
          &delta,
      const StaticSpan<SIZE, const Input> &a_previous,
      const StaticSpan<SIZE, const typename Derived::template Cache<Input>>
          &z_cache) const {
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
