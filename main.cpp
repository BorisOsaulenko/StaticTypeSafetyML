#include "include/mnist/TensorMnistReader.hpp"
#include "src/Network.h"
#include "src/activation/ActivationFunction.h"
#include "src/layers/Convolutional.h"
#include "src/layers/Dense.h"
#include "src/layers/Flatten.h"
#include "src/layers/MaxPool2x2.h"

template <size_t R> size_t argmax(const Vector<R> &vec) {
  size_t max_index = 0;
  double max_value = vec[0];

  for (size_t i = 1; i < R; ++i) {
    if (vec[i] > max_value) {
      max_value = vec[i];
      max_index = static_cast<int>(i);
    }
  }

  return max_index;
}

int main() {
  AdamFactory a(0.01);
  constexpr int train = 30000;
  constexpr int test = 5000;

  TensorMnistReader<train, test> reader;
  const auto train_set = reader.read_train_images();
  const auto validation_set = reader.read_test_images();

  using Input = Tensor3D<1, 28, 28>;
  using Output = Vector<10>;

  const StaticSpan<train, const Input> train_input(train_set.first);
  const StaticSpan<train, const Output> train_expected(train_set.second);
  const StaticSpan<test, const Input> validation_input(validation_set.first);
  const StaticSpan<test, const Output> validation_expected(
      validation_set.second);

  auto conv1 = Convolutional<1, 5, 5, 8>(ActivationFunctionEnum::Sigmoid, a);
  auto pool1 = MaxPool2x2();
  auto conv2 = Convolutional<8, 5, 5, 16>(ActivationFunctionEnum::Sigmoid, a);
  auto pool2 = MaxPool2x2();
  auto flatten = Flatten();
  auto dense1 = DenseLayer<16 * 4 * 4, 128>(ActivationFunctionEnum::Sigmoid, a);
  auto dense2 = DenseLayer<128, 10>(ActivationFunctionEnum::Identity, a);
  const std::string filename = "network_mnist_cnn.json";

  auto network = Network(LossEnum::CrossEntropy, conv1, pool1, conv2, pool2,
                         flatten, dense1, dense2);
  network.load(filename);
  std::cout << "Starting the training\n";
  network.train<400>(train_input, train_expected, validation_input,
                     validation_expected, 4, &std::cout);
  network.save(filename);

  const auto &train_predicted = network.predict_vectorized(train_input);
  const auto &test_predicted = network.predict_vectorized(validation_input);

  int correct = 0;
  for (int i = 0; i < train; ++i) {
    if (argmax(train_predicted[i]) == argmax(train_expected[i]))
      correct++;
  }

  std::cout << "Train accuracy: " << correct / static_cast<double>(train)
            << std::endl;

  correct = 0;
  for (int i = 0; i < test; ++i) {
    if (argmax(test_predicted[i]) == argmax(validation_expected[i]))
      correct++;
  }

  std::cout << "Test accuracy: " << correct / static_cast<double>(test)
            << std::endl;

  return 0;
}