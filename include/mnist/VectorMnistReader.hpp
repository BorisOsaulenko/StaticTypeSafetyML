#pragma once

#include "../../src/matrix/AlgebraicArray.h"
#include "../../src/matrix/Matrix.h"
#include "MnistReaderBase.hpp"

#include <vector>

#define PIXELS 784

template <int train_count, int test_count>
class VectorMnistReader : public MnistReaderBase<train_count, test_count> {
public:
  using Base = MnistReaderBase<train_count, test_count>;
  using Image = Vector<PIXELS>;
  using Label = Vector<10>;

  VectorMnistReader(std::string train_images_file,
                    std::string train_labels_file, std::string test_images_file,
                    std::string test_labels_file)
      : Base(std::move(train_images_file), std::move(train_labels_file),
             std::move(test_images_file), std::move(test_labels_file)) {}

  VectorMnistReader() = default;

  std::pair<AlgebraicArray<train_count, Image>,
            AlgebraicArray<train_count, Label>>
  read_train_images() {
    return read_images<train_count>(this->train_images_stream,
                                    this->train_labels_stream);
  }

  std::pair<AlgebraicArray<test_count, Image>,
            AlgebraicArray<test_count, Label>>
  read_test_images() {
    return read_images<test_count>(this->test_images_stream,
                                   this->test_labels_stream);
  }

private:
  template <int amount>
  static std::pair<AlgebraicArray<amount, Image>, AlgebraicArray<amount, Label>>
  read_images(std::ifstream &images_stream, std::ifstream &labels_stream) {
    AlgebraicArray<amount, Image> X;
    AlgebraicArray<amount, Label> y;
    uint8_t pixel;
    uint8_t label;

    for (int i = 0; i < amount; ++i) {
      Image xi;
      for (int j = 0; j < PIXELS; ++j) {
        images_stream.read(reinterpret_cast<char *>(&pixel), 1);
        xi[j] = static_cast<double>(pixel) / 255.0;
      }

      Label yi;
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