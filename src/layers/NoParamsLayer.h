//
// Created by Boris Osaulenko on 05/07/2026.
//

#ifndef ML_LAYER_H
#define ML_LAYER_H
#include "../matrix/Algebraic.h"
#include "../matrix/AlgebraicArray.h"
#include "../matrix/AlgebraicSpan.h"

#include <nlohmann/json.hpp>

template <Algebraic Input, Algebraic Output, std::size_t batch_size>
class NoParamsLayer {
  using json = nlohmann::json;

public:
  static constexpr bool has_params = false;
  explicit NoParamsLayer() : a_previous_cache(), z_cache() {}

  virtual ~NoParamsLayer() = default;

  AlgebraicArray<batch_size, Output>
  forward_vectorized(const AlgebraicSpan<batch_size, const Input> &inp) {
    this->a_previous_cache = inp;
    AlgebraicArray<batch_size, Output> z;

#pragma omp parallel for
    for (std::size_t i = 0; i < inp.size(); ++i) {
      z[i] = compute_z(inp[i]);
      z_cache[i] = z[i];
    }

    return z;
  }

  Output forward_no_cache(const Input &inp) const { return compute_z(inp); }

  template <std::size_t SIZE>
  AlgebraicArray<SIZE, Output>
  forward_no_cache_vectorized(const AlgebraicSpan<SIZE, Input> &inp) const {
    AlgebraicArray<SIZE, Output> result;

#pragma omp parallel for
    for (std::size_t i = 0; i < inp.size(); ++i) {
      result[i] = compute_z(inp[i]);
    }

    return result;
  }

  AlgebraicArray<batch_size, Input>
  backward_vectorized(const AlgebraicSpan<batch_size, const Output> &delta) {
    const int size = delta.size();
    AlgebraicArray<batch_size, Input> new_delta;

#pragma omp parallel for
    for (std::size_t i = 0; i < size; ++i) {
      new_delta[i] = compute_delta(delta[i], a_previous_cache[i]);
    }
    return new_delta;
  }

protected:
  using value_type = double;
  AlgebraicSpan<batch_size, const Input> a_previous_cache;
  AlgebraicArray<batch_size, Output> z_cache;

  virtual Output compute_z(const Input &inp) const = 0;
  virtual Input compute_delta(const Output &dz,
                              const Input &a_previous) const = 0;
};

#endif // ML_LAYER_H
