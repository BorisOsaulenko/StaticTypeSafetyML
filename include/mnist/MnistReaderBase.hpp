#pragma once

#include "../../src/matrix/Matrix.h"

#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#define MNIST_MAGIC_NUMBER_IMAGES 2051
#define MNIST_MAGIC_NUMBER_LABELS 2049

template <int train_count, int test_count> class MnistReaderBase {
protected:
  std::string train_images_file = "../datasets/mnist/train-images-idx3-ubyte";
  std::string train_labels_file = "../datasets/mnist/train-labels-idx1-ubyte";
  std::string test_images_file = "../datasets/mnist/t10k-images-idx3-ubyte";
  std::string test_labels_file = "../datasets/mnist/t10k-labels-idx1-ubyte";

  std::ifstream train_images_stream;
  std::ifstream train_labels_stream;
  std::ifstream test_images_stream;
  std::ifstream test_labels_stream;

  explicit MnistReaderBase(
      std::string train_images_file =
          "../datasets/mnist/train-images-idx3-ubyte",
      std::string train_labels_file =
          "../datasets/mnist/train-labels-idx1-ubyte",
      std::string test_images_file = "../datasets/mnist/t10k-images-idx3-ubyte",
      std::string test_labels_file = "../datasets/mnist/t10k-labels-idx1-ubyte")
      : train_images_file(std::move(train_images_file)),
        train_labels_file(std::move(train_labels_file)),
        test_images_file(std::move(test_images_file)),
        test_labels_file(std::move(test_labels_file)) {
    open_files();
    validate_opened_files();

    validate_images(train_images_stream, train_count);
    validate_labels(train_labels_stream, train_count);
    validate_images(test_images_stream, test_count);
    validate_labels(test_labels_stream, test_count);
  }

  static uint32_t read_u32(std::istream &in) {
    uint8_t bytes[4];
    in.read(reinterpret_cast<char *>(bytes), 4);

    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) |
           static_cast<uint32_t>(bytes[3]);
  }

  void open_files() {
    train_images_stream.open(train_images_file, std::ios::binary);
    train_labels_stream.open(train_labels_file, std::ios::binary);
    test_images_stream.open(test_images_file, std::ios::binary);
    test_labels_stream.open(test_labels_file, std::ios::binary);
  }

  void validate_opened_files() const {
    if (!train_images_stream.is_open())
      throw std::runtime_error("Could not open train images file");
    if (!train_labels_stream.is_open())
      throw std::runtime_error("Could not open train labels file");
    if (!test_images_stream.is_open())
      throw std::runtime_error("Could not open test images file");
    if (!test_labels_stream.is_open())
      throw std::runtime_error("Could not open test labels file");
  }

  static void validate_images(std::ifstream &images,
                              const unsigned int amount) {
    const uint32_t image_magic = read_u32(images);
    const uint32_t image_count = read_u32(images);
    const uint32_t rows = read_u32(images);
    const uint32_t cols = read_u32(images);

    if (image_magic != MNIST_MAGIC_NUMBER_IMAGES)
      throw std::runtime_error("Invalid image file.");

    if (rows != 28 || cols != 28)
      throw std::runtime_error("Unexpected image dimensions.");

    if (image_count < amount)
      throw std::runtime_error("Not enough images.");
  }

  static void validate_labels(std::ifstream &labels,
                              const unsigned int amount) {
    const uint32_t label_magic = read_u32(labels);
    const uint32_t label_count = read_u32(labels);

    if (label_magic != MNIST_MAGIC_NUMBER_LABELS)
      throw std::runtime_error("Invalid label file.");

    if (label_count < amount)
      throw std::runtime_error("Not enough labels.");
  }

  template <size_t SIZE, typename T>
  static void shuffle_data(StaticArray<SIZE, T> &X,
                           StaticArray<SIZE, Vector<10>> &y) {
    std::vector<int> indices(static_cast<int>(X.size()));
    for (size_t i = 0; i < indices.size(); ++i)
      indices[i] = static_cast<int>(i);

    std::random_device rd;
    std::mt19937 g(rd());
    std::ranges::shuffle(indices, g);

    StaticArray<SIZE, T> X_shuffled;
    StaticArray<SIZE, Vector<10>> y_shuffled;

    size_t i = 0;
    for (int idx : indices) {
      X_shuffled[i] = X[idx];
      y_shuffled[i] = y[idx];
      i++;
    }

    X = std::move(X_shuffled);
    y = std::move(y_shuffled);
  }
};
