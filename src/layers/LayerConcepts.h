
//
// Created by Boris Osaulenko on 14/08/2026.
//

#ifndef ML_LAYERCONCEPTS_H
#define ML_LAYERCONCEPTS_H
#include "../matrix/Algebraic.h"

#include <concepts>
#include <utility>

template <typename Input, Algebraic Grad> struct LayerBackwardResult {
  Input new_delta;
  Grad gradient;
};

template <typename T, typename Input>
concept ExposesCacheType = requires { typename T::template Cache<Input>; };

template <typename T, typename Input>
  requires ExposesCacheType<T, Input>
using Cache = T::template Cache<Input>;

template <typename T, typename Input>
concept ExposesOutputType = requires { typename T::template Output<Input>; };

template <typename T, typename Input>
  requires ExposesOutputType<T, Input>
using Output = T::template Output<Input>;

template <typename T, typename Input>
concept ComputesZNotCached = requires(const T &t, const Input &inp) {
  { t.compute_z_not_cached(inp) } -> std::same_as<Output<T, Input>>;
};

template <typename T, typename Input> struct LayerForwardCachedResult {
  Output<T, Input> output;
  Cache<T, Input> cache;
};

template <typename T, typename Input>
concept ComputesZCached = requires(const T &t, const Input &inp) {
  {
    t.compute_z_cached(inp)
  } -> std::same_as<LayerForwardCachedResult<T, Input>>;
};

template <typename R, typename Input>
struct IsLayerBackwardResultForImpl : std::false_type {};

template <typename Input, typename Grad>
struct IsLayerBackwardResultForImpl<LayerBackwardResult<Input, Grad>, Input>
    : std::true_type {};

template <typename R, typename Input>
concept BackwardResultFor = IsLayerBackwardResultForImpl<R, Input>::value;

template <typename T, typename Input>
concept ComputesDelta = requires(const T &t, const Input &inp) {
  {
    t.compute_delta(std::declval<Output<T, Input>>(), inp,
                    std::declval<Cache<T, Input>>())
  } -> BackwardResultFor<Input>;
};

template <typename T, typename Input>
concept ParamsLayerImpl =
    ExposesCacheType<T, Input> && ExposesOutputType<T, Input> &&
    ComputesZCached<T, Input> && ComputesZNotCached<T, Input> &&
    ComputesDelta<T, Input>;

template <typename T, typename Input>
concept ComputesDeltaDirect = requires(const T &t, const Input &inp) {
  {
    t.compute_delta(std::declval<Output<T, Input>>(), inp,
                    std::declval<Cache<T, Input>>())
  } -> std::same_as<Input>;
};

template <typename T, typename Input>
concept NoParamsLayerImpl =
    ExposesCacheType<T, Input> && ExposesOutputType<T, Input> &&
    ComputesZCached<T, Input> && ComputesDeltaDirect<T, Input>;

#endif // ML_LAYERCONCEPTS_H
