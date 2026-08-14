//
// Created by Boris Osaulenko on 16/07/2026.
//

#include "../../src/layers/MaxPool2x2.h"
#include "../../src/matrix/3DTensor.h"
#include "../../src/matrix/Matrix.h"
#include "utils.h"
#include <gtest/gtest.h>

constexpr double kTolerance = 0.05;

class MaxPool2x2Test : public testing::Test {
public:
  using Input = Tensor3D<2, 4, 4>;
  using Output = Tensor3D<2, 2, 2>;
  static const inline auto input = StaticArray<1, Input>({Input{
      Matrix<4, 4>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      Matrix<4, 4>{16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}}});

  static inline auto delta_item =
      Output{Matrix<2, 2>{1, 2, 3, 4}, Matrix<2, 2>{-1, 0, 2, 1}};
  static const inline auto delta = StaticArray<1, Output>({delta_item});

  static inline StaticArray<1, Output> z_cache{};

protected:
  MaxPool2x2<2, 4, 4> max_pool_layer{};

  void SetUp() override {}
};

TEST_F(MaxPool2x2Test, ForwardPass) {
  const Tensor3D<2, 2, 2> expected = {Matrix<2, 2>{6, 8, 14, 16},
                                      Matrix<2, 2>{16, 14, 8, 6}};

  const auto output = max_pool_layer.forward_vectorized(
      StaticSpan<1, const Input>(input), z_cache);
  expect_tensor_close<2, 2, 2>(output[0], expected, kTolerance);
  expect_tensor_close<2, 2, 2>(z_cache[0], expected, kTolerance);
}

TEST_F(MaxPool2x2Test, BackwardPass) {
  max_pool_layer.forward_vectorized(StaticSpan<1, const Input>(input),
                                    z_cache);

  const Tensor3D<2, 4, 4> expected = {
      Matrix<4, 4>{0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 3, 0, 4},
      Matrix<4, 4>{-1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0}};

  const auto new_delta =
      max_pool_layer.backward_vectorized<1>(delta, input, z_cache);
  expect_tensor_close<2, 4, 4>(new_delta[0], expected, kTolerance);
}