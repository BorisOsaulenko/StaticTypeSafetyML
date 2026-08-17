
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

template <typename T, typename Input, typename Output>
concept ComputesZ = requires(const T &t, const Input &inp) {
  { t.compute_z(inp) } -> std::same_as<Output>;
};

template <typename T, typename Input, typename Output, typename Grad>
concept ComputesDelta = requires(const T &t, const Output &delta,
                                 const Input &inp, const Output &z_cache) {
  {
    t.compute_delta(delta, inp, z_cache)
  } -> std::same_as<LayerBackwardResult<Input, Grad>>;
};

template <typename T, typename Input, typename Output, typename Grad>
concept ParamsLayerImpl =
    ComputesZ<T, Input, Output> && ComputesDelta<T, Input, Output, Grad>;

template <typename T, typename Input, typename Output>
concept ComputesDeltaDirect = requires(
    const T &t, const Output &delta, const Input &inp, const Output &z_cache) {
  { t.compute_delta(delta, inp, z_cache) } -> std::same_as<Input>;
};

template <typename T, typename Input, typename Output>
concept NoParamsLayerImpl =
    ComputesZ<T, Input, Output> && ComputesDeltaDirect<T, Input, Output>;

// --- Layer-composition checks, used to validate a chain of layers duck-typed
// on compute_z/compute_delta rather than on a single nominal Input/Output
// pair (a layer's compute_z may be generic, e.g. templated on a runtime-sized
// dimension, so it need not expose one fixed Output type).

template <typename Layer, typename In>
concept AcceptsInput = requires(const Layer &l, const In &in) {
  { l.compute_z(in) };
};

template <typename Layer, typename In>
using ComputeZResult = decltype(std::declval<const Layer &>().compute_z(
    std::declval<const In &>()));

// compute_delta returns either Input directly (NoParamsLayer) or
// LayerBackwardResult<Input, Grad> (ParamsLayer); extract the "new_delta"
// type either way so forward and backward chains can be compared uniformly.
template <typename T> struct NewDeltaExtract {
  using type = T;
};
template <typename Input, typename Grad>
struct NewDeltaExtract<LayerBackwardResult<Input, Grad>> {
  using type = Input;
};
template <typename T> using new_delta_t = typename NewDeltaExtract<T>::type;

template <typename Layer, typename In, typename Out>
using ComputeDeltaResult = decltype(std::declval<const Layer &>().compute_delta(
    std::declval<const Out &>(), std::declval<const In &>(),
    std::declval<const Out &>()));

template <typename Layer, typename In, typename Out>
using NewDeltaResult = new_delta_t<ComputeDeltaResult<Layer, In, Out>>;

// A layer's compute_delta must accept what its own compute_z produces, and
// must hand back a gradient wrt Input that is itself In-shaped -- which is
// exactly what the next layer upstream needs to receive as its own
// incoming delta during backprop, so this one per-layer check is enough to
// guarantee the whole backward chain lines up (no separate backward pass
// needed).
template <typename Layer, typename In, typename Out>
concept BackwardConsistent =
    requires(const Layer &l, const Out &delta, const In &a_prev, const Out &z) {
      { l.compute_delta(delta, a_prev, z) };
    } && std::same_as<NewDeltaResult<Layer, In, Out>, In>;

#endif // ML_LAYERCONCEPTS_H
