#ifndef ML_MODEL_H
#define ML_MODEL_H
#include "layers/Dense.h"
#include "loss/Loss.h"
#include "loss/LossFactory.h"
#include "matrix/AlgebraicSpan.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

template <int I, typename L1> consteval bool CheckLayers() { return true; }

template <int I, typename L1, typename L2, typename... Rest>
consteval bool CheckLayers() {
  static_assert(std::is_same_v<typename L1::Output, typename L2::Input>,
                "Layers are incompatible");

  return CheckLayers<I + 1, L2, Rest...>();
}

template <std::size_t N, typename F> constexpr void constexpr_for(F &&f) {
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    (f(std::integral_constant<std::size_t, Is>{}), ...);
  }(std::make_index_sequence<N>{});
}

template <typename... Layers> class Network {
  static_assert(CheckLayers<0, Layers...>());

  using json = nlohmann::json;
  using Input = Layers...[0] ::Input;
  using Output = Layers...[sizeof...(Layers) - 1] ::Output;

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
  [[nodiscard]] AlgebraicArray<SIZE, Output>
  predict_vectorized(const AlgebraicSpan<SIZE, Input> inputs) const {
    return forward_no_cache_vectorized<0>(inputs);
  }

  template <std::size_t batch_size, std::size_t train_size,
            std::size_t validation_size>
  void
  train(const AlgebraicSpan<train_size, const Input> train_inputs,
        const AlgebraicSpan<train_size, const Output> train_expected,
        const AlgebraicSpan<validation_size, const Input> validation_inputs,
        const AlgebraicSpan<validation_size, const Output> validation_expected,
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
  mean_loss(const AlgebraicSpan<SIZE, const Input> &inputs,
            const AlgebraicSpan<SIZE, const Output> &expected,
            const int batch_size = 1000) {
    double total_loss = 0.0;
    const std::size_t sample_count = std::min<std::size_t>(SIZE, batch_size);
    const int n = static_cast<int>(sample_count);

#pragma omp parallel for reduction(+ : total_loss) schedule(static)
    for (int i = 0; i < n; ++i) {
      const Output predicted = forward_no_cache<0>(inputs[i]);
      total_loss += loss->compute(predicted, expected[i]);
    }

    return total_loss / static_cast<double>(sample_count);
  }

  [[nodiscard]] LossEnum get_loss_type() const { return loss_type; }

  friend void to_json(json &j, const Network &network) {
    json layers_array = json::array();
    to_json_recursive<0>(layers_array, network.layers);
    j = json{{"layers", layers_array}, {"loss_type", network.loss_type}};
  }

  friend void from_json(const json &j, Network &network) {
    const auto &layers_json = j.at("layers");

    if (!layers_json.is_array() || layers_json.size() != sizeof...(Layers))
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

  template <std::size_t I>
  static void from_json_recursive(const json &layers_json,
                                  std::tuple<Layers &...> &layers) {
    if constexpr (I < sizeof...(Layers)) {
      if (auto &layer = std::get<I>(layers); layer.has_params) {
        layers_json.at(I).get_to(layer);
      }
      from_json_recursive<I + 1>(layers_json, layers);
    }
  }

private:
  std::tuple<Layers &...> layers;
  LossEnum loss_type;
  std::unique_ptr<Loss<Output>> loss;

  std::mt19937 rng;
  std::uniform_real_distribution<> dist;

  template <std::size_t batch, std::size_t train_size, std::size_t batch_size>
  void train_batch(const AlgebraicSpan<train_size, const Input> train_inputs,
                   const AlgebraicSpan<train_size, const Output> train_expected,
                   std::ostream *log_stream = nullptr) {
    const AlgebraicSpan<batch_size, const Input> batch_inputs =
        train_inputs
            .template subspan<batch * batch_size, (batch + 1) * batch_size>();
    const AlgebraicSpan<batch_size, const Output> batch_expected =
        train_expected
            .template subspan<batch * batch_size, (batch + 1) * batch_size>();

    const AlgebraicArray<batch_size, Output> predicted =
        forward_vectorized<0, Input>(batch_inputs);
    const AlgebraicArray<batch_size, Output> delta =
        loss->derivative((predicted), batch_expected);
    backward_vectorized<sizeof...(Layers) - 1, Output>(delta);

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

  template <std::size_t I, typename Tuple, typename T, std::size_t SIZE>
  auto
  forward_no_cache_vectorized(Tuple &tpl,
                              const AlgebraicSpan<SIZE, const T> inputs) const {
    if constexpr (I == std::tuple_size_v<Tuple> - 1) {
      return std::get<I>(tpl).forward_no_cache_vectorized(inputs);
    } else {
      using LayerOut =
          std::remove_reference_t<std::tuple_element_t<I, Tuple>>::Output;
      auto &layer = std::get<I>(tpl);
      const auto out = layer.forward_no_cache_vectorized(inputs);
      return forward_no_cache_vectorized<I + 1>(
          tpl, AlgebraicSpan<SIZE, const LayerOut>(out));
    }
  }

  template <std::size_t I, typename Inp, std::size_t SIZE>
  auto forward_vectorized(const AlgebraicSpan<SIZE, const Inp> inputs) {
    using LayerOut = std::remove_reference_t<Layers...[I]>::Output;
    auto &layer = std::get<I>(layers);

    if constexpr (I == sizeof...(Layers) - 1) {
      return layer.forward_vectorized(inputs);
    } else {
      const auto out = layer.forward_vectorized(inputs);
      return forward_vectorized<I + 1>(
          AlgebraicSpan<SIZE, const LayerOut>(out));
    }
  }

  template <std::size_t I, typename DeltaType, std::size_t SIZE>
  void backward_vectorized(const AlgebraicArray<SIZE, DeltaType> &delta) {
    using Layer = Layers...[I];
    auto &layer = std::get<I>(layers);

    if constexpr (I == 0) {
      layer.backward_vectorized(delta);
    } else {
      auto new_delta = layer.backward_vectorized(delta);
      backward_vectorized<I - 1, typename Layer::Input, SIZE>(new_delta);
    }
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
      const AlgebraicSpan<train_size, const Input> train_inputs,
      const AlgebraicSpan<train_size, const Output> train_expected,
      const AlgebraicSpan<validation_size, const Input> validation_inputs,
      const AlgebraicSpan<validation_size, const Output> validation_expected) {
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
