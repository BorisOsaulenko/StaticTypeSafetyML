
//
// Created by Boris Osaulenko on 14/08/2026.
//

#ifndef ML_LAYERCONCEPTS_H
#define ML_LAYERCONCEPTS_H
#include "../matrix/Algebraic.h"
#include "../matrix/StaticArray.h"

#include <concepts>
#include <string>
#include <utility>

template <typename NewDelta, typename Grad> struct BackwardResult {
  NewDelta new_delta;
  Grad gradient;
};

template <typename Out, typename CacheT> struct ForwardCachedResult {
  Out output;
  CacheT cache;
};

template <typename R> struct IsAnyForwardCachedResult : std::false_type {};
template <typename Out, typename CacheT>
struct IsAnyForwardCachedResult<ForwardCachedResult<Out, CacheT>>
    : std::true_type {};
template <typename R>
concept AnyForwardCachedResult = IsAnyForwardCachedResult<R>::value;

template <typename T, typename Input>
concept ComputesZ = requires(const T &t, const Input &inp) {
  { t.compute_z(inp) };
  { t.compute_z_cached(inp) } -> AnyForwardCachedResult;
  // cached and non-cached paths must agree on the output type
  requires std::same_as<decltype(t.compute_z_cached(inp).output),
                        decltype(t.compute_z(inp))>;
};

template <typename T, typename Input>
  requires ComputesZ<T, Input>
using LayerOutput = decltype(std::declval<const T &>().compute_z(
    std::declval<const Input &>()));

template <typename T, typename Input>
  requires ComputesZ<T, Input>
using LayerCache = decltype(std::declval<const T &>()
                                .compute_z_cached(std::declval<const Input &>())
                                .cache);

template <typename T, typename Input>
  requires ComputesZ<T, Input>
using LayerForwardCachedResult =
    ForwardCachedResult<LayerOutput<T, Input>, LayerCache<T, Input>>;

template <typename T, typename Input, typename Grad>
concept ComputesDeltaCached =
    requires(const T &t, const LayerOutput<T, Input> &delta,
             const LayerCache<T, Input> &cache) {
      {
        t.compute_delta(delta, cache)
      } -> std::same_as<BackwardResult<Input, Grad>>;
    };

template <typename T, typename Input>
concept ComputesDelta = requires(const T &t, const LayerOutput<T, Input> &delta,
                                 const LayerCache<T, Input> &cache) {
  { t.compute_delta(delta, cache) } -> std::same_as<Input>;
};

template <typename T, typename Input, typename Grad>
concept ParamsLayerImpl = Algebraic<Grad> && ComputesZ<T, Input> &&
                          ComputesDeltaCached<T, Input, Grad>;

template <typename T, typename Input>
concept NoParamsLayerImpl = ComputesZ<T, Input> && ComputesDelta<T, Input>;

template <typename T> consteval std::string_view type_name() {
  std::string_view fn = __PRETTY_FUNCTION__;
  const std::size_t begin = fn.find("T = ") + 4;
  const std::size_t end = fn.find_first_of(";]", begin);
  return fn.substr(begin, end - begin);
}

template <typename L> constexpr std::string_view LayerToString = type_name<L>();

#endif // ML_LAYERCONCEPTS_H
