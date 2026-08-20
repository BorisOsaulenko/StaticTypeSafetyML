#ifndef ML_MODEL_H
#define ML_MODEL_H
#include "NetworkConcepts.h"
#include "layers/Dense.h"
#include "loss/Loss.h"
#include "loss/LossFactory.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>

template <int I, typename In, typename L1> consteval bool CheckLayers() {
  static_assert(AcceptsInput<L1, In>,
                "Layer does not accept the previous layer's output");
  static_assert(BackwardConsistent<L1, In, ComputeZResult<L1, In>>,
                "Layer's compute_delta is inconsistent with its compute_z");
  return true;
}

template <int I, typename In, typename L1, typename L2, typename... Rest>
consteval bool CheckLayers() {
  static_assert(AcceptsInput<L1, In>,
                "Layer does not accept the previous layer's output");
  static_assert(BackwardConsistent<L1, In, ComputeZResult<L1, In>>,
                "Layer's compute_delta is inconsistent with its compute_z");

  return CheckLayers<I + 1, ComputeZResult<L1, In>, L2, Rest...>();
}

template <std::size_t N, typename F> constexpr void constexpr_for(F &&f) {
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    (f(std::integral_constant<std::size_t, Is>{}), ...);
  }(std::make_index_sequence<N>{});
}

template <typename... Layers> class Network {
  static_assert(CheckLayers<0, typename Layers...[0] ::Input, Layers...>());

  using json = nlohmann::json;
  using LayerTuple = std::tuple<Layers...>;
  using Input = Layers...[0] ::Input;
  using Output = Layers...[sizeof...(Layers) - 1] ::Output;
  template <std::size_t SIZE, typename T>
  using LayerCacheArray = StaticArray<SIZE, T>;

  template <std::size_t SIZE>
  using LayerInputCaches =
      std::tuple<LayerCacheArray<SIZE, typename Layers::Input>...>;

  template <std::size_t SIZE>
  using LayerOutputCaches =
      std::tuple<LayerCacheArray<SIZE, typename Layers::Output>...>;

public:
  explicit Network(const LossEnum loss_type, Layers &...layers)
      : layers(layers...), loss_type(loss_type),
        loss(LossFactory::create<Output>(loss_type)) {
    rng.seed(std::random_device{}());
  }

  [[nodiscard]] Output predict(const Input &input) const {
    return forward_no_cache<0>(input);
  }

  template <std::size_t SIZE>
  [[nodiscard]] StaticArray<SIZE, Output>
  predict_vectorized(const StaticSpan<SIZE, const Input> inputs) const {
    return forward_no_cache_vectorized<0>(inputs);
  }

  template <std::size_t batch_size, std::size_t train_size,
            std::size_t validation_size>
  void
  train(const StaticSpan<train_size, const Input> train_inputs,
        const StaticSpan<train_size, const Output> train_expected,
        const StaticSpan<validation_size, const Input> validation_inputs,
        const StaticSpan<validation_size, const Output> validation_expected,
        const int &epochs = 10, std::ostream *log_stream = nullptr)
    requires(train_size % batch_size == 0)
  {
    constexpr std::size_t batch_amount = train_size / batch_size;

    for (long long epoch = 0; epoch < epochs; ++epoch) {
      constexpr_for<batch_amount>([&](auto i) {
        train_batch<i, train_size, batch_size>(train_inputs, train_expected,
                                               log_stream);
      });
      log_epoch(log_stream, epoch, train_inputs, train_expected,
                validation_inputs, validation_expected);
    }
  }

  void save(const std::string &filename) const {
    json j;
    to_json(j, *this);
    std::ofstream file(filename);
    if (!file.is_open())
      throw std::runtime_error("Could not open file for writing: " + filename);

    file << j.dump(2);
    file.close();
  }

  void load(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open())
      throw std::runtime_error("Could not open file for reading: " + filename);

    json j;
    file >> j;
    file.close();
    from_json(j, *this);
  }

  template <std::size_t SIZE>
  [[nodiscard]] double
  mean_loss(const StaticSpan<SIZE, const Input> &inputs,
            const StaticSpan<SIZE, const Output> &expected) {
    const StaticArray<SIZE, Output> predictions = predict_vectorized(inputs);
    const Vector<SIZE> losses = loss->compute(predictions, expected);

    return losses.sum() / SIZE;
  }

  [[nodiscard]] LossEnum get_loss_type() const { return loss_type; }

  friend void to_json(json &j, const Network &network) {
    json layers_array = json::array();
    to_json_recursive<0>(layers_array, network.layers);
    j = json{{"layers", layers_array}, {"loss_type", network.loss_type}};
  }

  friend void from_json(const json &j, Network &network) {
    const auto &layers_json = j.at("layers");

    if (!layers_json.is_array())
      throw std::invalid_argument("Invalid layer data in network JSON.");

    from_json_recursive<0>(layers_json, network.layers);

    j.at("loss_type").get_to(network.loss_type);
    network.loss = LossFactory::create<Output>(network.loss_type);
  }

  template <std::size_t I>
  static void to_json_recursive(json &layers_array,
                                const std::tuple<Layers &...> &layers) {
    if constexpr (I < sizeof...(Layers)) {
      if constexpr (auto &layer = std::get<I>(layers); layer.has_params) {
        json layer_json;
        layer_json = layer;
        layers_array.push_back(layer_json);
      }

      to_json_recursive<I + 1>(layers_array, layers);
    }
  }

  template <std::size_t I, std::size_t A = 0>
  static void from_json_recursive(const json &layers_json,
                                  std::tuple<Layers &...> &layers) {
    if constexpr (I < sizeof...(Layers)) {
      if constexpr (auto &layer = std::get<I>(layers); layer.has_params) {
        layers_json.at(A).get_to(layer);
        return from_json_recursive<I + 1, A + 1>(layers_json, layers);
      }
      from_json_recursive<I + 1, A>(layers_json, layers);
    }
  }

private:
  std::tuple<Layers &...> layers;
  LossEnum loss_type;
  std::unique_ptr<Loss<Output>> loss;

  std::mt19937 rng;
  std::uniform_real_distribution<> dist;

  template <std::size_t batch, std::size_t train_size, std::size_t batch_size>
  void train_batch(const StaticSpan<train_size, const Input> train_inputs,
                   const StaticSpan<train_size, const Output> train_expected,
                   std::ostream *log_stream = nullptr) {
    const StaticSpan<batch_size, const Input> batch_inputs =
        train_inputs
            .template subspan<batch * batch_size, (batch + 1) * batch_size>();
    const StaticSpan<batch_size, const Output> batch_expected =
        train_expected
            .template subspan<batch * batch_size, (batch + 1) * batch_size>();

    LayerInputCaches<batch_size> a_previous_cache;
    LayerOutputCaches<batch_size> z_cache;

    const StaticArray<batch_size, Output> predicted =
        forward_vectorized<0, Input>(a_previous_cache, z_cache, batch_inputs);
    const StaticArray<batch_size, Output> delta =
        loss->derivative((predicted), batch_expected);
    backward_vectorized<sizeof...(Layers) - 1, Output>(delta, a_previous_cache,
                                                       z_cache);

    update_layers<0>(batch_size);
    if (log_stream)
      *log_stream << "Batch " << batch << " finished." << std::endl;
  }

  template <std::size_t I, typename T> auto forward_no_cache(T &&input) const {
    if constexpr (I == sizeof...(Layers)) {
      return std::forward<T>(input);
    } else {
      auto &layer = std::get<I>(layers);
      auto out = layer.forward_no_cache(std::forward<T>(input));
      return forward_no_cache<I + 1>(std::move(out));
    }
  }

  template <std::size_t I, typename T, std::size_t SIZE>
  auto
  forward_no_cache_vectorized(const StaticSpan<SIZE, const T> inputs) const {
    auto &layer = std::get<I>(layers);
    const auto out = layer.forward_no_cache_vectorized(inputs);

    if constexpr (I == sizeof...(Layers) - 1)
      return out;
    else
      return forward_no_cache_vectorized<I + 1>(StaticSpan(out));
  }

  template <std::size_t I, typename Inp, std::size_t SIZE>
  auto forward_vectorized(LayerInputCaches<SIZE> &a_previous_cache,
                          LayerOutputCaches<SIZE> &z_cache,
                          const StaticArray<SIZE, Inp> inputs) {
    auto &layer = std::get<I>(layers);
    std::get<I>(a_previous_cache) = inputs;
    const auto out = layer.forward_vectorized(
        StaticSpan<SIZE, const Inp>(inputs), std::get<I>(z_cache));

    if constexpr (I == sizeof...(Layers) - 1) {
      return out;
    } else {
      return forward_vectorized<I + 1>(a_previous_cache, z_cache, out);
    }
  }

  template <std::size_t I, typename Inp, std::size_t SIZE>
  auto forward_vectorized(LayerInputCaches<SIZE> &a_previous_cache,
                          LayerOutputCaches<SIZE> &z_cache,
                          const StaticSpan<SIZE, const Inp> inputs) {
    auto &layer = std::get<I>(layers);
    std::get<I>(a_previous_cache) = copy_span(inputs);
    const auto out = layer.forward_vectorized(inputs, std::get<I>(z_cache));

    if constexpr (I == sizeof...(Layers) - 1) {
      return out;
    } else {
      return forward_vectorized<I + 1>(a_previous_cache, z_cache, out);
    }
  }

  template <std::size_t SIZE, typename T>
  StaticArray<SIZE, T> copy_span(const StaticSpan<SIZE, const T> &span) {
    StaticArray<SIZE, T> result;
    for (std::size_t i = 0; i < SIZE; ++i) {
      result[i] = span[i];
    }
    return result;
  }

  template <std::size_t I, typename DeltaType, std::size_t SIZE>
  void backward_vectorized(const StaticArray<SIZE, DeltaType> &delta,
                           const LayerInputCaches<SIZE> &a_previous_cache,
                           const LayerOutputCaches<SIZE> &z_cache) {
    using Layer = Layers...[I];
    auto &layer = std::get<I>(layers);
    using LayerInput = Layer::Input;
    using LayerOutput = Layer::Output;

    [[maybe_unused]] const auto new_delta = layer.backward_vectorized(
        StaticSpan<SIZE, const LayerOutput>(delta),
        StaticSpan<SIZE, const LayerInput>(std::get<I>(a_previous_cache)),
        StaticSpan<SIZE, const LayerOutput>(std::get<I>(z_cache)));

    if constexpr (I != 0)
      backward_vectorized<I - 1, LayerInput, SIZE>(new_delta, a_previous_cache,
                                                   z_cache);
  }

  template <std::size_t I> auto update_layers(std::size_t batch_remainder) {
    if constexpr (I != sizeof...(Layers)) {
      if constexpr (auto &layer = std::get<I>(layers); layer.has_params) {
        layer.apply_change(batch_remainder);
      }
      update_layers<I + 1>(batch_remainder);
    }
  }

  template <std::size_t train_size, std::size_t validation_size>
  void log_epoch(
      std::ostream *log_stream, const int epoch,
      const StaticSpan<train_size, const Input> train_inputs,
      const StaticSpan<train_size, const Output> train_expected,
      const StaticSpan<validation_size, const Input> validation_inputs,
      const StaticSpan<validation_size, const Output> validation_expected) {
    if (log_stream) {
      *log_stream << "Epoch: " << epoch << std::endl;
      *log_stream << "-------------------" << std::endl;
      *log_stream << "Training loss: "
                  << mean_loss(train_inputs, train_expected) << std::endl;

      if (validation_size != 0)
        *log_stream << "Validation loss: "
                    << mean_loss(validation_inputs, validation_expected)
                    << std::endl;
      else
        *log_stream << "Validation loss: skipped.";
      *log_stream << std::endl;
    }
  }
};

#endif // ML_MODEL_H
