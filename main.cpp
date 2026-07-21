//
// Created by Boris Osaulenko on 13/07/2026.
//

#include "include/mnist/TensorMnistReader.hpp"
#include "include/mnist/VectorMnistReader.hpp"
#include "src/Network.h"
#include "src/layers/Convolutional.h"
#include "src/layers/Dense.h"
#include "src/layers/Flatten.h"
#include "src/layers/MaxPool2x2.h"
#include "src/lr_optimizer/Adam.h"

int main() {
  AdamFactory a(0.001);
  constexpr int train = 10;
  constexpr int test = 2;
  constexpr std::size_t batch_size = 10;

  TensorMnistReader<train, test> reader;
  const auto train_set = reader.read_train_images();
  const auto validation_set = reader.read_test_images();

  using Input = Tensor3D<1, 28, 28>;
  using Output = Vector<10>;

  const AlgebraicSpan<train, const Input> train_input(train_set.first);
  const AlgebraicSpan<train, const Output> train_expected(train_set.second);
  const AlgebraicSpan<test, const Input> validation_input(validation_set.first);
  const AlgebraicSpan<test, const Output> validation_expected(
      validation_set.second);

  auto conv1 = Convolutional<1, 28, 28, 5, 5, 8, batch_size>(
      ActivationFunctionEnum::Sigmoid, a);
  auto pool1 = MaxPool2x2<8, 24, 24, batch_size>();
  auto conv2 = Convolutional<8, 12, 12, 5, 5, 16, batch_size>(
      ActivationFunctionEnum::Sigmoid, a);
  auto pool2 = MaxPool2x2<16, 8, 8, batch_size>();
  auto flatten = Flatten<16, 4, 4, batch_size>();
  auto dense1 = DenseLayer<16 * 4 * 4, 128, batch_size>(
      ActivationFunctionEnum::Sigmoid, a);
  auto dense2 =
      DenseLayer<128, 10, batch_size>(ActivationFunctionEnum::Identity, a);
  auto noop = Convolutional<1, 28, 28, 1, 1, 1, batch_size>(
      ActivationFunctionEnum::Identity, a);
  const std::string filename = "network_mnist_cnn.json";

  auto network = Network(LossEnum::CrossEntropy, conv1, pool1, conv2, pool2,
                         flatten, dense1, dense2);
  // network.load(filename);
  std::cout << "Starting the training\n";
  network.train<batch_size>(train_input, train_expected, validation_input,
                            validation_expected, 800, &std::cout);
  network.save(filename);

  // int correct = 0;
  // for (int i = 0; i < train; ++i) {
  //   auto input = train_input[i];
  //   auto expected = train_expected[i];
  //
  //   Vector<10> prediction = network.predict(input);
  //   const auto argmax =
  //       std::distance(prediction.begin(),
  //       std::ranges::max_element(prediction));
  //   const auto expected_label =
  //       std::distance(expected.begin(), std::ranges::max_element(expected));
  //
  //   if (argmax == expected_label)
  //     correct++;
  // }
  //
  // std::cout << "Train accuracy: " << correct / static_cast<double>(train)
  //           << std::endl;
  //
  // correct = 0;
  // for (int i = 0; i < test; ++i) {
  //   auto input = test_input[i];
  //   auto expected = test_expected[i];
  //
  //   Vector<10> prediction = network.predict(input);
  //   const auto argmax =
  //       std::distance(prediction.begin(),
  //       std::ranges::max_element(prediction));
  //   const auto expected_label =
  //       std::distance(expected.begin(), std::ranges::max_element(expected));
  //
  //   if (argmax == expected_label)
  //     correct++;
  // }
  //
  // std::cout << "Test accuracy: " << correct / static_cast<double>(test)
  //           << std::endl;
  //
  // return 0;
}