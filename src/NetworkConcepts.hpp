#ifndef ML_NETWORKCONCEPTS_H
#define ML_NETWORKCONCEPTS_H
#include "layers/LayerConcepts.hpp"
#include "layers/NoParamsLayer.hpp"
#include "layers/ParamsLayer.hpp"
#include <string.h>

template <typename L> constexpr bool isParamsLayer() {
  if constexpr (requires { typename L::Params; })
    return std::derived_from<L, ParamsLayer<L, typename L::Params>>;
  else
    return false;
}

template <typename L> constexpr bool isNoParamsLayer() {
  return std::derived_from<L, NoParamsLayer<L>>;
}

template <typename In, typename... Ls> struct NetworkOutput {
  using type = In;
}; // no layers left: the input is the output

template <typename In, typename L, typename... Rest>
struct NetworkOutput<In, L, Rest...> {
  using type = typename NetworkOutput<LayerOutput<L, In>, Rest...>::type;
};

template <typename In, typename... Ls>
using NetworkOutputT = typename NetworkOutput<In, Ls...>::type;

constexpr std::string to_str(size_t n) {
  std::string s;
  do {
    s.insert(s.begin(), char('0' + n % 10));
    n /= 10;
  } while (n);
  return s;
}

template <size_t I, typename In, typename L1> consteval bool CheckLayers() {
  static_assert(ComputesZ<L1, In>,
                "Last layer does not accept the pre-last layer's output");
  static_assert(
      (isNoParamsLayer<L1>() && ComputesDelta<L1, In>) ||
          (isParamsLayer<L1>() &&
           ComputesDeltaCached<L1, In, typename L1::Params>),
      "Last layer's compute_delta is inconsistent with its compute_z");
  return true;
}

template <size_t I, typename In, typename L1, typename L2, typename... Rest>
consteval bool CheckLayers() {
  static_assert(ComputesZ<L1, In>,
                "Layer " + to_str(I) +
                    " does not accept the previous layer's output");
  if constexpr (isParamsLayer<L1>()) {
    static_assert(ComputesDeltaCached<L1, In, typename L1::Params>,
                  "Params Layer " + to_str(I) +
                      " compute delta is inconsistent with its compute_z");
  } else {
    static_assert(ComputesDelta<L1, In>,
                  "NoParams Layer's " + to_str(I) +
                      " compute_delta is inconsistent with its compute_z");
  }
  return CheckLayers<I + 1, LayerOutput<L1, In>, L2, Rest...>();
};

// Ith layer input by the network input.
template <size_t I, size_t Target, typename Input, typename L1,
          typename... Rest>
struct LayerIInput
    : LayerIInput<I + 1, Target, LayerOutput<L1, Input>, Rest...> {};

template <size_t Target, typename Input, typename L1, typename... Rest>
struct LayerIInput<Target, Target, Input, L1, Rest...> {
  using type = Input;
};

template <size_t Target, typename Input, typename... Layers>
  requires(Target < sizeof...(Layers))
using LayerInputT = typename LayerIInput<0, Target, Input, Layers...>::type;

// Ith layer output by the layers input.
template <size_t I, size_t Target, typename Input, typename L1,
          typename... Rest>
struct LayerIOutput
    : LayerIOutput<I + 1, Target, LayerOutput<L1, Input>, Rest...> {};

template <size_t Target, typename Input, typename L1, typename... Rest>
struct LayerIOutput<Target, Target, Input, L1, Rest...> {
  using type = LayerOutput<L1, Input>;
};

template <size_t Target, typename Input, typename... Layers>
  requires(Target < sizeof...(Layers))
using LayerOutputT = typename LayerIOutput<0, Target, Input, Layers...>::type;

#endif // ML_NETWORKCONCEPTS_H
