#ifndef ML_LAYER_H
#define ML_LAYER_H
#include "../matrix/StaticArray.h"
#include "LayerConcepts.h"
#include <concepts>

#define SIZE_INP_TEMPLATE template <size_t SIZE, typename Input>
#define REQUIRES_IMPL requires NoParamsLayerImpl<Derived, Input>
#define OUTPUT LayerOutput<Derived, Input>
#define CACHE LayerCache<Derived, Input>

#define CACHE_ERROR                                                            \
  std::string("Cache has invalid type for ") + LayerToString<Derived> +        \
      ". Should be StaticArray<SIZE, LayerCache<Derived, Input> &"

#define DELTA_ERROR                                                            \
  std::string("Delta has invalid type for ") + LayerToString<Derived> +        \
      ". Should be StaticSpan<SIZE, LayerOutput<Derived, Input>> &"

template <typename Derived> class NoParamsLayer {
  const Derived &derived() const { return static_cast<const Derived &>(*this); }

  template <typename Input> struct InputContainer {};

public:
  static constexpr bool has_params = false;
  explicit NoParamsLayer() = default;
  NoParamsLayer(const NoParamsLayer &other) = default;

  template <typename Input>
  auto forward_no_cache(const Input &inp) const REQUIRES_IMPL {
    return this->derived().compute_z(inp);
  }

  SIZE_INP_TEMPLATE
  auto forward_vectorized(const StaticArray<SIZE, Input> &inp,
                          auto &cache) const REQUIRES_IMPL {
    return forward_vectorized(StaticSpan<SIZE, const Input>(inp), cache);
  }

  SIZE_INP_TEMPLATE
  auto forward_vectorized(const StaticSpan<SIZE, const Input> &inp,
                          auto &cache) const REQUIRES_IMPL {
    static_assert(std::same_as<StaticArray<SIZE, CACHE> &, decltype(cache)>,
                  CACHE_ERROR);
    StaticArray<SIZE, OUTPUT> result;

#pragma omp parallel for
    for (size_t i = 0; i < inp.size(); ++i) {
      LayerForwardCachedResult<Derived, Input> r =
          this->derived().compute_z_cached(inp[i]);
      result[i] = r.output;
      cache[i] = r.cache;
    }

    return result;
  }

  SIZE_INP_TEMPLATE
  auto forward_no_cache_vectorized(
      const StaticSpan<SIZE, const Input> &inp) const REQUIRES_IMPL {
    StaticArray<SIZE, OUTPUT> result;

#pragma omp parallel for
    for (size_t i = 0; i < inp.size(); ++i) {
      result[i] = this->derived().compute_z(inp[i]);
    }

    return result;
  }

  SIZE_INP_TEMPLATE
  StaticArray<SIZE, Input>
  backward_vectorized(const auto &delta,
                      const auto &cache) const REQUIRES_IMPL {
    // const StaticArray<100, StaticArray<8, Matrix<12, 12> > >&
    // const StaticSpan<100, const StaticArray<8, Matrix<12, 12> > &
    static_assert(
        std::same_as<const StaticArray<SIZE, OUTPUT> &, decltype(delta)>,
        DELTA_ERROR);
    static_assert(
        std::same_as<const StaticArray<SIZE, CACHE> &, decltype(cache)>,
        CACHE_ERROR);
    StaticArray<SIZE, Input> new_delta;

#pragma omp parallel for
    for (size_t i = 0; i < delta.size(); ++i) {
      new_delta[i] = this->derived().compute_delta(delta[i], cache[i]);
    }
    return new_delta;
  }
};

#undef SIZE_INP_TEMPLATE
#undef REQUIRES_IMPL
#undef OUTPUT
#undef CACHE
#undef CACHE_ERROR
#undef DELTA_ERROR

#endif // ML_LAYER_H
