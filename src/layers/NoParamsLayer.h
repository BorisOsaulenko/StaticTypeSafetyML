//
// Created by Boris Osaulenko on 05/07/2026.
//

#ifndef ML_LAYER_H
#define ML_LAYER_H
#include "../matrix/Algebraic.h"
#include "../matrix/StaticArray.h"

#include <nlohmann/json.hpp>

template <typename Input, Algebraic Output> class NoParamsLayer {
  using json = nlohmann::json;

public:
  static constexpr bool has_params = false;
  explicit NoParamsLayer() = default;

  virtual ~NoParamsLayer() = default;

  template <std::size_t SIZE>
  StaticArray<SIZE, Output>
  forward_vectorized(const StaticSpan<SIZE, const Input> &inp,
                     StaticArray<SIZE, Output> &z_cache) const {
    StaticArray<SIZE, Output> z;

#pragma omp parallel for
    for (std::size_t i = 0; i < inp.size(); ++i) {
      z[i] = compute_z(inp[i]);
      z_cache[i] = z[i];
    }

    return z;
  }

  Output forward_no_cache(const Input &inp) const { return compute_z(inp); }

  template <std::size_t SIZE>
  StaticArray<SIZE, Output>
  forward_no_cache_vectorized(const StaticSpan<SIZE, const Input> &inp) const {
    StaticArray<SIZE, Output> result;

#pragma omp parallel for
    for (std::size_t i = 0; i < inp.size(); ++i) {
      result[i] = compute_z(inp[i]);
    }

    return result;
  }

  template <std::size_t SIZE>
  StaticArray<SIZE, Input>
  backward_vectorized(const StaticSpan<SIZE, const Output> &delta,
                      const StaticSpan<SIZE, const Input> &a_previous,
                      const StaticSpan<SIZE, const Output> &z_cache) const {
    StaticArray<SIZE, Input> new_delta;

#pragma omp parallel for
    for (std::size_t i = 0; i < delta.size(); ++i) {
      new_delta[i] = compute_delta(delta[i], a_previous[i], z_cache[i]);
    }
    return new_delta;
  }

protected:
  using value_type = double;
  virtual Output compute_z(const Input &inp) const = 0;
  virtual Input compute_delta(const Output &delta, const Input &a_previous,
                              const Output &z_cache) const = 0;
};

#endif // ML_LAYER_H
