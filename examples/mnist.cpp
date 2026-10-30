#include "../include/mnist/VectorMnistReader.hpp"
#include "../src/Network.hpp"
#include "../src/layers/Dense.hpp"
#include "../src/lr_optimizer/Adam.hpp"
#include <cstddef>
#include <random>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// its technically possible that a vector has more than 1 entry with highest
// probability, but it can be ignored.
template <size_t R> size_t argmax(const Vector<R> &vec) {
  double max_val = vec[0];
  size_t max_idx = 0;
  for (size_t i = 1; i < R; ++i) {
    if (vec[i] > max_val) {
      max_val = vec[i];
      max_idx = i;
    }
  }
  return max_idx;
}

template <size_t R, size_t BATCH>
size_t correct_count(const StaticArray<BATCH, Vector<R>> &predictions,
                     const StaticArray<BATCH, Vector<R>> &expected) {
  size_t correct = 0;
  for (size_t i = 0; i < BATCH; ++i) {
    const size_t predicted_label = argmax(predictions[i]);
    const size_t expected_label = argmax(expected[i]);

    if (predicted_label == expected_label)
      correct++;
  }

  return correct;
}

int main() {
  AdamFactory optimizer_factory(0.01);
  constexpr int TRAIN = 10000;
  constexpr int TEST = 2000;
  constexpr int BATCH = 200;

  VectorMnistReader<TRAIN, TEST> reader;
  auto [train_input, train_expected] = reader.read_train_images();
  auto [test_input, test_expected] = reader.read_test_images();
  auto layer1 = DenseLayer<784, 128>(ActivationFunctionEnum::ReLu);
  auto layer2 = DenseLayer<128, 10>(ActivationFunctionEnum::Identity);
  const std::string filename = "network_mnist.json";

  auto network =
      Network(optimizer_factory, LossEnum::CrossEntropy, layer1, layer2);
  // network.load(filename);

  std::cout << "Starting the training.\n";
  network.train<BATCH, TRAIN, TEST, Vector<784>>(
      train_input, train_expected, test_input, test_expected, 8, &std::cout);
  network.save(filename);

  const StaticArray<TRAIN, Vector<10>> &predictions_train =
      network.predict_vectorized(train_input);

  size_t correct = correct_count(predictions_train, train_expected);
  std::cout << "Train accuracy: " << correct / static_cast<double>(TRAIN)
            << std::endl;

  const StaticArray<TEST, Vector<10>> predictions_test =
      network.predict_vectorized(test_input);
  correct = correct_count(predictions_test, test_expected);

  std::cout << "Test accuracy: " << correct / static_cast<double>(TEST)
            << std::endl;

  std::mt19937 gen(std::random_device{}());
  std::vector<int> picked;
  std::ranges::sample(std::views::iota(0, TEST), std::back_inserter(picked), 6,
                      gen);

  for (const int idx : picked) {
    std::array<unsigned char, 280 * 280> px{}; // 28x28 scaled up 10x
    for (int y = 0; y < 280; ++y)
      for (int x = 0; x < 280; ++x)
        px[y * 280 + x] = static_cast<unsigned char>(
            test_input[idx][(y / 10) * 28 + x / 10] * 255.0);

    const std::string file =
        std::format("predicted_{}.png", argmax(predictions_test[idx]));
    stbi_write_png(file.c_str(), 280, 280, 1, px.data(), 280);
    std::system(("open " + file).c_str());
  }

  return 0;
}