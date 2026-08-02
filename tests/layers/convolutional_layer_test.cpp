//
// Created by Boris Osaulenko on 15/07/2026.
//

#include "../../src/layers/Convolutional.h"
#include "utils.h"
#include <gtest/gtest.h>

constexpr double kTolerance = 0.05;

class ConvolutionalTest : public testing::Test {
public:
  using Input = Tensor3D<2, 3, 3>;
  using Output = Tensor3D<2, 2, 2>;
  using Kernel = Tensor3D<2, 2, 2>;
  using KernelArray = AlgebraicArray<2, Kernel>;
  using Conv = Convolutional<2, 3, 3, 2, 2, 2>;

  static const inline Input input = {Matrix<3, 3>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                                     Matrix<3, 3>{9, 8, 7, 6, 5, 4, 3, 2, 1}};

  static const inline Output delta = {Matrix<2, 2>{1, 2, 3, 4},
                                      Matrix<2, 2>{-1, 0, 2, 1}};

protected:
  AdamFactory optimizer_factory;

  static const inline Kernel kernel1 = {Matrix<2, 2>{1, 0, 0, 1},
                                        Matrix<2, 2>{0, 1, 1, 0}};
  static const inline Kernel kernel2 = {Matrix<2, 2>{1, 1, 1, 1},
                                        Matrix<2, 2>{-1, -1, -1, -1}};
  static const inline KernelArray kernels = {kernel1, kernel2};
  static const inline Vector<2> bias = {1, -1};

  Conv::Params params{kernels, bias};
  Conv convolutional_layer{params, ActivationFunctionEnum::Identity,
                           optimizer_factory};
};

TEST_F(ConvolutionalTest, ForwardPass) {
  const AlgebraicArray<1, Input> inputs{input};
  AlgebraicArray<1, Output> z_cache;

  const auto output = convolutional_layer.forward_vectorized<1>(
      AlgebraicSpan<1, const Input>(inputs), z_cache);

  const Output expected = {Matrix<2, 2>{21, 21, 21, 21},
                           Matrix<2, 2>{-17, -9, 7, 15}};

  expect_tensor_close<2, 2, 2>(output[0], expected, kTolerance);
  expect_tensor_close<2, 2, 2>(z_cache[0], expected, kTolerance);
}

TEST_F(ConvolutionalTest, BackwardPassCalculatesKernelGradients) {
  const AlgebraicArray<1, Input> inputs{input};
  const AlgebraicArray<1, Output> deltas{delta};
  AlgebraicArray<1, Output> z_cache;

  convolutional_layer.forward_vectorized<1>(
      AlgebraicSpan<1, const Input>(inputs), z_cache);
  convolutional_layer.backward_vectorized<1>(
      AlgebraicSpan<1, const Output>(deltas),
      AlgebraicSpan<1, const Input>(inputs), z_cache);

  const Kernel expected_kernel_gradients1 = {Matrix<2, 2>{37, 47, 67, 77},
                                             Matrix<2, 2>{63, 53, 33, 23}};
  const Kernel expected_kernel_gradients2 = {Matrix<2, 2>{12, 14, 18, 20},
                                             Matrix<2, 2>{8, 6, 2, 0}};

  const auto kernel_gradients = convolutional_layer.get_kernel_gradients();
  expect_matrix_close<2, 2>(kernel_gradients[0][0],
                            expected_kernel_gradients1[0], kTolerance);
  expect_matrix_close<2, 2>(kernel_gradients[0][1],
                            expected_kernel_gradients1[1], kTolerance);
  expect_matrix_close<2, 2>(kernel_gradients[1][0],
                            expected_kernel_gradients2[0], kTolerance);
  expect_matrix_close<2, 2>(kernel_gradients[1][1],
                            expected_kernel_gradients2[1], kTolerance);
}

TEST_F(ConvolutionalTest, BackwardPassCalculatesBiasGradients) {
  const AlgebraicArray<1, Input> inputs{input};
  const AlgebraicArray<1, Output> deltas{delta};
  AlgebraicArray<1, Output> z_cache;

  convolutional_layer.forward_vectorized<1>(
      AlgebraicSpan<1, const Input>(inputs), z_cache);
  convolutional_layer.backward_vectorized<1>(
      AlgebraicSpan<1, const Output>(deltas),
      AlgebraicSpan<1, const Input>(inputs), z_cache);

  const Vector<2> expected_bias_gradients = {10, 2};
  expect_vector_close<2>(convolutional_layer.get_bias_gradients(),
                         expected_bias_gradients, kTolerance);
}

TEST_F(ConvolutionalTest, BackwardPassReturnsNewDelta) {
  const AlgebraicArray<1, Input> inputs{input};
  const AlgebraicArray<1, Output> deltas{delta};
  AlgebraicArray<1, Output> z_cache;

  convolutional_layer.forward_vectorized<1>(
      AlgebraicSpan<1, const Input>(inputs), z_cache);
  const auto new_delta = convolutional_layer.backward_vectorized<1>(
      AlgebraicSpan<1, const Output>(deltas),
      AlgebraicSpan<1, const Input>(inputs), z_cache);

  const Input expected = {Matrix<3, 3>{0, 1, 0, 4, 7, 3, 2, 6, 5},
                          Matrix<3, 3>{1, 2, 2, 0, 3, 3, 1, 1, -1}};

  expect_tensor_close<2, 3, 3>(new_delta[0], expected, kTolerance);
}
