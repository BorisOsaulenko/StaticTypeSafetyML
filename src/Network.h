#ifndef ML_MODEL_H
#define ML_MODEL_H
#include "NetworkConcepts.h"
#include "layers/LayerConcepts.h"
#include "loss/Loss.h"
#include "loss/LossFactory.h"
#include "matrix/AlgebraicTuple.h"
#include "matrix/Matrix.h"
#include "matrix/StaticArray.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <tuple>
#include <utility>

#define REQUIRES_LAYER_CONSISTENCY requires(CheckLayers<0, Input, Layers...>())
#define OUTPUT NetworkOutputT<Input, Layers...>
#define ITH_LAYER_OUTPUT LayerOutputT<I, Input, Layers...>
#define ITH_LAYER_INPUT LayerInputT<I, Input, Layers...>

template <size_t N, typename F> constexpr void constexpr_for(F &&f) {
  [&]<size_t... Is>(std::index_sequence<Is...>) {
    (f(std::integral_constant<size_t, Is>{}), ...);
  }(std::make_index_sequence<N>{});
}

template <typename... Layers> class Network {
  using json = nlohmann::json;
  using LayerTuple = std::tuple<Layers...>;

  template <size_t SIZE, typename Input, size_t... Is>
  static auto make_layer_caches(std::index_sequence<Is...>)
      -> AlgebraicTuple<StaticArray<
          SIZE, LayerCache<Layers, LayerInputT<Is, Input, Layers...>>>...>;

  template <size_t SIZE, typename Input>
  using LayerCaches = decltype(make_layer_caches<SIZE, Input>(
      std::index_sequence_for<Layers...>{}));

public:
  explicit Network(const LossEnum loss_type, Layers &...layers)
      : layers(layers...), loss_type(loss_type) {
    rng.seed(std::random_device{}());
  }

  template <typename Input>
  [[nodiscard]] OUTPUT
  predict(const Input &input) const REQUIRES_LAYER_CONSISTENCY {
    return forward_no_cache<0>(input);
  }

  template <size_t SIZE, typename Input>
  [[nodiscard]] StaticArray<SIZE, OUTPUT> predict_vectorized(
      const StaticArray<SIZE, Input> inputs) const REQUIRES_LAYER_CONSISTENCY {
    return forward_no_cache_vectorized<0, SIZE, Input>(inputs);
  };

  template <size_t SIZE, typename Input>
  [[nodiscard]] StaticArray<SIZE, OUTPUT>
  predict_vectorized(const StaticSpan<SIZE, const Input> inputs) const
      REQUIRES_LAYER_CONSISTENCY {
    return forward_no_cache_vectorized<0, SIZE, Input>(inputs);
  }

  template <size_t BATCH_SZ, size_t TRAIN_SZ, size_t VALIDATION_SZ,
            typename Input>
  void train(const StaticSpan<TRAIN_SZ, const Input> train_inputs,
             const StaticSpan<TRAIN_SZ, const OUTPUT> train_expected,
             const StaticSpan<VALIDATION_SZ, const Input> validation_inputs,
             const StaticSpan<VALIDATION_SZ, const OUTPUT> validation_expected,
             const int &epochs = 10, std::ostream *log_stream = nullptr)
    requires(TRAIN_SZ % BATCH_SZ == 0)
  {
    constexpr size_t AmtBatch = TRAIN_SZ / BATCH_SZ;

    for (long long epoch = 0; epoch < epochs; ++epoch) {
      constexpr_for<AmtBatch>([&](auto i) {
        train_batch<i, TRAIN_SZ, BATCH_SZ>(train_inputs, train_expected,
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

  template <size_t SIZE, typename Input>
  [[nodiscard]] double
  mean_loss(const StaticSpan<SIZE, const Input> &inputs,
            const StaticSpan<SIZE, const OUTPUT> &expected) {
    std::unique_ptr<Loss<OUTPUT>> loss = LossFactory::create<OUTPUT>(loss_type);
    const StaticArray<SIZE, OUTPUT> predictions = predict_vectorized(inputs);
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
  }

private:
  std::tuple<Layers &...> layers;
  LossEnum loss_type;

  std::mt19937 rng;
  std::uniform_real_distribution<> dist;

  template <size_t I>
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

  template <size_t I, size_t A = 0>
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

  template <size_t CURR_BATCH, size_t TRAIN_SZ, size_t BATCH_SZ, typename Input>
  void train_batch(const StaticSpan<TRAIN_SZ, const Input> train_inputs,
                   const StaticSpan<TRAIN_SZ, const OUTPUT> train_expected,
                   std::ostream *log_stream = nullptr) {
    auto start = std::chrono::steady_clock::now();
    const StaticSpan<BATCH_SZ, const Input> batch_inputs =
        train_inputs.template subspan<CURR_BATCH * BATCH_SZ,
                                      (CURR_BATCH + 1) * BATCH_SZ>();
    const StaticSpan<BATCH_SZ, const OUTPUT> batch_expected =
        train_expected.template subspan<CURR_BATCH * BATCH_SZ,
                                        (CURR_BATCH + 1) * BATCH_SZ>();

    LayerCaches<BATCH_SZ, Input> caches;
    std::unique_ptr<Loss<OUTPUT>> loss = LossFactory::create<OUTPUT>(loss_type);

    const StaticArray<BATCH_SZ, OUTPUT> predicted =
        forward_vectorized<0, Input, BATCH_SZ>(batch_inputs, caches);
    const StaticArray<BATCH_SZ, OUTPUT> delta =
        loss->derivative((predicted), batch_expected);
    backward_vectorized<sizeof...(Layers) - 1, Input, BATCH_SZ>(delta, caches);

    update_layers<0>(BATCH_SZ);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end - start;
    if (log_stream)
      *log_stream << "Batch " << CURR_BATCH
                  << " finished. Duration: " << duration << std::endl;
  }

  template <size_t I, typename T> auto forward_no_cache(T &&input) const {
    if constexpr (I == sizeof...(Layers)) {
      return std::forward<T>(input);
    } else {
      auto &layer = std::get<I>(layers);
      auto out = layer.forward_no_cache(std::forward<T>(input));
      return forward_no_cache<I + 1>(std::move(out));
    }
  }

  template <size_t I, size_t SIZE, typename Input>
  StaticArray<SIZE, OUTPUT> forward_no_cache_vectorized(
      const StaticSpan<SIZE, const LayerInputT<I, Input, Layers...>> &inputs)
      const {
    auto &layer = std::get<I>(layers);
    const auto out = layer.forward_no_cache_vectorized(inputs);

    if constexpr (I == sizeof...(Layers) - 1)
      return out;
    else
      return forward_no_cache_vectorized<I + 1, SIZE, Input>(StaticSpan(out));
  }

  template <size_t I, typename Input, size_t SIZE>
  StaticArray<SIZE, OUTPUT>
  forward_vectorized(const StaticArray<SIZE, ITH_LAYER_INPUT> inputs,
                     LayerCaches<SIZE, Input> &cache) {
    return forward_vectorized<I, Input, SIZE>(
        StaticSpan<SIZE, const ITH_LAYER_INPUT>(inputs), cache);
  }

  template <size_t I, typename Input, size_t SIZE>
  StaticArray<SIZE, OUTPUT>
  forward_vectorized(const StaticSpan<SIZE, const ITH_LAYER_INPUT> inputs,
                     LayerCaches<SIZE, Input> &cache) {
    auto &layer = std::get<I>(layers);

    const StaticArray<SIZE, ITH_LAYER_OUTPUT> cachedOutput =
        layer.forward_vectorized(
            StaticSpan<SIZE, const ITH_LAYER_INPUT>(inputs),
            cache.template get<I>());

    if constexpr (I == sizeof...(Layers) - 1) {
      return cachedOutput;
    } else {
      return forward_vectorized<I + 1, Input, SIZE>(cachedOutput, cache);
    }
  }

  template <size_t SIZE, typename T>
  StaticArray<SIZE, T> copy_span(const StaticSpan<SIZE, const T> &span) {
    StaticArray<SIZE, T> result;
    for (size_t i = 0; i < SIZE; ++i) {
      result[i] = span[i];
    }
    return result;
  }

  template <size_t I, typename Input, size_t SIZE>
  void backward_vectorized(const StaticArray<SIZE, ITH_LAYER_OUTPUT> &delta,
                           const LayerCaches<SIZE, Input> &cache) {
    auto &layer = std::get<I>(layers);

    [[maybe_unused]] const StaticArray<SIZE, ITH_LAYER_INPUT> new_delta =
        layer.template backward_vectorized<SIZE, ITH_LAYER_INPUT>(
            delta, cache.template get<I>());

    if constexpr (I != 0)
      backward_vectorized<I - 1, Input, SIZE>(new_delta, cache);
  }

  template <size_t I> auto update_layers(size_t batch_remainder) {
    if constexpr (I != sizeof...(Layers)) {
      if constexpr (auto &layer = std::get<I>(layers); layer.has_params) {
        layer.apply_change(batch_remainder);
      }
      update_layers<I + 1>(batch_remainder);
    }
  }

  template <size_t train_size, size_t validation_size, typename Input>
  void log_epoch(
      std::ostream *log_stream, const int epoch,
      const StaticSpan<train_size, const Input> train_inputs,
      const StaticSpan<train_size, const OUTPUT> train_expected,
      const StaticSpan<validation_size, const Input> validation_inputs,
      const StaticSpan<validation_size, const OUTPUT> validation_expected) {
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

#undef REQUIRES_LAYER_CONSISTENCY
#undef OUTPUT
#undef ITH_LAYER_OUTPUT
#undef ITH_LAYER_INPUT

#endif // ML_MODEL_H
