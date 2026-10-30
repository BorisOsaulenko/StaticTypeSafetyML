#include "../../src/layers/Convolutional.hpp"
#include "../utils.hpp"
#include <gtest/gtest.h>

constexpr double kTolerance = 0.05;

class ConvolutionalTest : public testing::Test {
public:
  using Input = Tensor3D<2, 3, 3>;
  using Output = Tensor3D<2, 2, 2>;
  using Kernel = Tensor3D<2, 2, 2>;
  using KernelArray = StaticArray<2, Kernel>;
  using Conv = Convolutional<2, 2, 2, 2>;

  static const inline Input input_elem = {
      Matrix<3, 3>{1, 2, 3, 4, 5, 6, 7, 8, 9},
      Matrix<3, 3>{9, 8, 7, 6, 5, 4, 3, 2, 1}};

  static const inline Output delta_elem = {Matrix<2, 2>{1, 2, 3, 4},
                                           Matrix<2, 2>{-1, 0, 2, 1}};

  static inline StaticArray<1, Input> input{input_elem};
  static inline StaticArray<1, Output> delta{delta_elem};

protected:
  AdamFactory optimizer_factory;

  static const inline Kernel kernel1 = {Matrix<2, 2>{1, 0, 0, 1},
                                        Matrix<2, 2>{0, 1, 1, 0}};
  static const inline Kernel kernel2 = {Matrix<2, 2>{1, 1, 1, 1},
                                        Matrix<2, 2>{-1, -1, -1, -1}};
  static const inline KernelArray kernels = {kernel1, kernel2};
  static const inline Vector<2> bias = {1, -1};

  Conv::Params params{kernels, bias};
  Conv convolutional_layer{params, ActivationFunctionEnum::Identity};
};

TEST_F(ConvolutionalTest, ForwardPass) {
  StaticArray<1, LayerCache<Conv, Input>> cache;

  const StaticArray<1, LayerOutput<Conv, Input>> output =
      convolutional_layer.forward_vectorized(input, cache);

  const LayerCache<Conv, Input> expectedCache = {
      input_elem, {Matrix<2, 2>{21, 21, 21, 21}, Matrix<2, 2>{-17, -9, 7, 15}}};
  const Output expectedOutput = {Matrix<2, 2>{21, 21, 21, 21},
                                 Matrix<2, 2>{-17, -9, 7, 15}};

  expect_tensor_close<2, 2, 2>(output[0], expectedCache.z_cache, kTolerance);
  expect_tensor_close<2, 2, 2>(cache[0].z_cache, expectedCache.z_cache,
                               kTolerance);
}

TEST_F(ConvolutionalTest, BackwardPassCalculatesKernelGradients) {

  const StaticArray<1, Output> deltas{delta_elem};
  StaticArray<1, LayerCache<Conv, Input>> z_cache;

  convolutional_layer.forward_vectorized<1, Input>(input, z_cache);
  convolutional_layer.backward_vectorized<1, Input>(deltas, z_cache);

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
  const StaticArray<1, Input> inputs{input_elem};
  const StaticArray<1, Output> deltas{delta_elem};
  StaticArray<1, LayerCache<Conv, Input>> cache;

  convolutional_layer.forward_vectorized<1, Input>(inputs, cache);
  convolutional_layer.backward_vectorized<1, Input>(deltas, cache);

  const Vector<2> expected_bias_gradients = {10, 2};
  expect_vector_close<2>(convolutional_layer.get_bias_gradients(),
                         expected_bias_gradients, kTolerance);
}

TEST_F(ConvolutionalTest, BackwardPassReturnsNewDelta) {
  const StaticArray<1, Input> inputs{input_elem};
  const StaticArray<1, Output> deltas{delta_elem};
  StaticArray<1, LayerCache<Conv, Input>> cache;

  convolutional_layer.forward_vectorized<1, Input>(inputs, cache);
  const auto new_delta =
      convolutional_layer.backward_vectorized<1, Input>(deltas, cache);

  const Input expected = {Matrix<3, 3>{0, 1, 0, 4, 7, 3, 2, 6, 5},
                          Matrix<3, 3>{1, 2, 2, 0, 3, 3, 1, 1, -1}};

  expect_tensor_close<2, 3, 3>(new_delta[0], expected, kTolerance);
}
