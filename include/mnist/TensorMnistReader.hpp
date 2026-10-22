#pragma once

#include "../../src/matrix/3DTensor.h"
#include "MnistReaderBase.hpp"

#include <vector>

template <int train_count, int test_count>
class TensorMnistReader : public MnistReaderBase<train_count, test_count> {
public:
  using Base = MnistReaderBase<train_count, test_count>;

  TensorMnistReader(std::string train_images_file,
                    std::string train_labels_file, std::string test_images_file,
                    std::string test_labels_file)
      : Base(std::move(train_images_file), std::move(train_labels_file),
             std::move(test_images_file), std::move(test_labels_file)) {}

  TensorMnistReader() = default;

  std::pair<StaticArray<train_count, Tensor3D<1, 28, 28>>,
            StaticArray<train_count, Vector<10>>>
  read_train_images() {
    return read_images<train_count>(this->train_images_stream,
                                    this->train_labels_stream);
  }

  std::pair<StaticArray<test_count, Tensor3D<1, 28, 28>>,
            StaticArray<test_count, Vector<10>>>
  read_test_images() {
    return read_images<test_count>(this->test_images_stream,
                                   this->test_labels_stream);
  }

private:
  template <int amount>
  static std::pair<StaticArray<amount, Tensor3D<1, 28, 28>>,
                   StaticArray<amount, Vector<10>>>
  read_images(std::ifstream &images_stream, std::ifstream &labels_stream) {
    StaticArray<amount, Tensor3D<1, 28, 28>> X;
    StaticArray<amount, Vector<10>> y;

    uint8_t pixel;
    uint8_t label;

    for (size_t i = 0; i < amount; ++i) {
      Tensor3D<1, 28, 28> xi;
      for (size_t r = 0; r < 28; ++r) {
        for (size_t c = 0; c < 28; ++c) {
          images_stream.read(reinterpret_cast<char *>(&pixel), 1);
          xi[0][r, c] = static_cast<double>(pixel) / 255.0;
        }
      }

      Vector<10> yi;
      for (int k = 0; k < 10; ++k)
        yi[k] = 0.0;
      labels_stream.read(reinterpret_cast<char *>(&label), 1);
      yi[label] = 1.0;

      X[i] = xi;
      y[i] = yi;
    }

    Base::shuffle_data(X, y);

    return {std::move(X), std::move(y)};
  }
};
