#include "../include/mnist/VectorMnistReader.hpp"
#include "../src/Network.h"
#include "../src/layers/Dense.h"
#include "../src/lr_optimizer/Adam.h"
#include <opencv2/opencv.hpp>
#include <random>

int main() {
  AdamFactory optimizer_factory(0.001);
  constexpr int train = 10000;
  constexpr int test = 2000;

  VectorMnistReader<train, test> reader;
  auto [train_input, train_expected] = reader.read_train_images();
  auto [test_input, test_expected] = reader.read_test_images();
  auto layer1 = DenseLayer<784, 128>(ActivationFunctionEnum::ReLu);
  auto layer2 = DenseLayer<128, 10>(ActivationFunctionEnum::Identity);
  const std::string filename = "network_mnist.json";

  auto network =
      Network(optimizer_factory, LossEnum::CrossEntropy, layer1, layer2);
  // network.load(filename);

  network.train(train_input, train_expected, test_input, test_expected, 8,
                &std::cout);
  network.save(filename);

  int correct = 0;
  for (int i = 0; i < train; ++i) {
    auto input = train_input[i];
    auto expected = train_expected[i];

    Vector<10> prediction = network.predict(input);
    const auto argmax =
        std::distance(prediction.begin(), std::ranges::max_element(prediction));
    const auto expected_label =
        std::distance(expected.begin(), std::ranges::max_element(expected));

    if (argmax == expected_label)
      correct++;
  }

  std::cout << "Train accuracy: " << correct / static_cast<double>(train)
            << std::endl;

  correct = 0;
  for (int i = 0; i < test; ++i) {
    auto input = test_input[i];
    auto expected = test_expected[i];

    Vector<10> prediction = network.predict(input);
    const auto argmax =
        std::distance(prediction.begin(), std::ranges::max_element(prediction));
    const auto expected_label =
        std::distance(expected.begin(), std::ranges::max_element(expected));

    if (argmax == expected_label)
      correct++;
  }

  std::cout << "Test accuracy: " << correct / static_cast<double>(test)
            << std::endl;

  return 0;
}