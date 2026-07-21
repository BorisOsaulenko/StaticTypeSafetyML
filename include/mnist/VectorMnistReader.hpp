#pragma once

#include "../../src/matrix/Matrix.h"
#include "MnistReaderBase.hpp"

#include <vector>

#define PIXELS 784

template <int train_count, int test_count>
class VectorMnistReader : public MnistReaderBase<train_count, test_count> {
public:
  using Base = MnistReaderBase<train_count, test_count>;

  VectorMnistReader(std::string train_images_file,
                    std::string train_labels_file, std::string test_images_file,
                    std::string test_labels_file)
      : Base(std::move(train_images_file), std::move(train_labels_file),
             std::move(test_images_file), std::move(test_labels_file)) {}

  VectorMnistReader() = default;

  std::pair<std::vector<Vector<PIXELS>>, std::vector<Vector<10>>>
  read_train_images() {
    return read_images<train_count>(this->train_images_stream,
                                    this->train_labels_stream);
  }

  std::pair<std::vector<Vector<PIXELS>>, std::vector<Vector<10>>>
  read_test_images() {
    return read_images<test_count>(this->test_images_stream,
                                   this->test_labels_stream);
  }

private:
  template <int amount>
  static std::pair<std::vector<Vector<PIXELS>>, std::vector<Vector<10>>>
  read_images(std::ifstream &images_stream, std::ifstream &labels_stream) {
    std::vector<Vector<PIXELS>> X;
    std::vector<Vector<10>> y;
    X.reserve(amount);
    y.reserve(amount);

    uint8_t pixel;
    uint8_t label;

    for (int i = 0; i < amount; ++i) {
      Vector<PIXELS> xi;
      for (int j = 0; j < PIXELS; ++j) {
        images_stream.read(reinterpret_cast<char *>(&pixel), 1);
        xi[j] = static_cast<double>(pixel) / 255.0;
      }

      Vector<10> yi;
      for (int k = 0; k < 10; ++k)
        yi[k] = 0.0;
      labels_stream.read(reinterpret_cast<char *>(&label), 1);
      yi[label] = 1.0;

      X.push_back(xi);
      y.push_back(yi);
    }

    Base::shuffle_data(X, y);

    return {std::move(X), std::move(y)};
  }
};