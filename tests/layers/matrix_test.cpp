#include "../../src/matrix/AlgebraicTuple.h"
#include "../../src/matrix/Matrix.h"
#include "utils.h"

#include <cmath>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

static constexpr double kTolerance = 1e-6;

namespace {

using Matrix2x2 = Matrix<2, 2>;
using Matrix3x3 = Matrix<3, 3>;
using Vector2 = Vector<2>;
using Tensor = Tensor3D<2, 2, 2>;
using PairTuple = AlgebraicTuple<Matrix2x2, Matrix2x2>;

} // namespace

class MatrixTest : public testing::Test {
public:
  const Matrix3x3 input{1, 2, 3, 4, 5, 6, 7, 8, 9};
  const Matrix2x2 kernel{1, 2, 3, 4};
  const Matrix2x2 other{5, 6, 7, 8};
  const Vector2 vector{2, 3};
};

TEST_F(MatrixTest, BasicIndexingAndArithmetic) {
  EXPECT_DOUBLE_EQ((input[0, 0]), 1);
  EXPECT_DOUBLE_EQ((input[2, 1]), 8);

  const auto sum = kernel + other;
  const auto diff = other - kernel;
  const auto quotient = other / kernel;

  expect_matrix_close<2, 2>(sum, Matrix2x2{6, 8, 10, 12}, kTolerance);
  expect_matrix_close<2, 2>(diff, Matrix2x2{4, 4, 4, 4}, kTolerance);
  expect_matrix_close<2, 2>(quotient, Matrix2x2{5, 3, 7.0 / 3.0, 2},
                            kTolerance);

  auto inplace = kernel;
  inplace += other;
  expect_matrix_close<2, 2>(inplace, Matrix2x2{6, 8, 10, 12}, kTolerance);
  inplace -= other;
  expect_matrix_close<2, 2>(inplace, kernel, kTolerance);
  inplace *= 2;
  expect_matrix_close<2, 2>(inplace, Matrix2x2{2, 4, 6, 8}, kTolerance);
  inplace /= 2;
  expect_matrix_close<2, 2>(inplace, kernel, kTolerance);
}

TEST_F(MatrixTest, ScalarAndElementwiseOperations) {
  expect_matrix_close<2, 2>(kernel + 1, Matrix2x2{2, 3, 4, 5}, kTolerance);
  expect_matrix_close<2, 2>(kernel - 1, Matrix2x2{0, 1, 2, 3}, kTolerance);
  expect_matrix_close<2, 2>(kernel * 2, Matrix2x2{2, 4, 6, 8}, kTolerance);
  expect_matrix_close<2, 2>(kernel / 2, Matrix2x2{0.5, 1, 1.5, 2}, kTolerance);

  expect_matrix_close<2, 2>(kernel.hadamard(other), Matrix2x2{5, 12, 21, 32},
                            kTolerance);
  EXPECT_DOUBLE_EQ(kernel.dot(other), 70);
  EXPECT_DOUBLE_EQ(kernel.sum(), 10);

  const auto squared =
      kernel.elementwise([](const double value) { return value * value; });
  expect_matrix_close<2, 2>(squared, Matrix2x2{1, 4, 9, 16}, kTolerance);

  const Matrix2x2 exp_input{0, 1, -1, 2};
  const auto exp_output = exp_input.exp();
  expect_matrix_close<2, 2>(
      exp_output,
      Matrix2x2{std::exp(0.0), std::exp(1.0), std::exp(-1.0), std::exp(2.0)},
      kTolerance);

  const Matrix2x2 sqrt_input{1, 4, 9, 16};
  expect_matrix_close<2, 2>(sqrt_input.sqrt(), Matrix2x2{1, 2, 3, 4},
                            kTolerance);

  const Matrix2x2 log_input{1, 2.718281828459045, 7.38905609893065,
                            20.085536923187668};
  expect_matrix_close<2, 2>(log_input.log(), Matrix2x2{0, 1, 2, 3}, kTolerance);
}

TEST_F(MatrixTest, MatrixSpecificOperations) {
  expect_matrix_close<2, 2>(input.valid_cross_correlation(kernel),
                            Matrix2x2{37, 47, 67, 77}, kTolerance);

  expect_matrix_close<4, 4>(
      input.full_convolution(kernel),
      Matrix<4, 4>{1, 4, 7, 6, 7, 23, 33, 24, 19, 53, 63, 42, 21, 52, 59, 36},
      kTolerance);

  expect_matrix_close<2, 2>(
      Matrix2x2::outer_product(Vector2{1, 2}, Vector2{2, 3}),
      Matrix2x2{2, 3, 4, 6}, kTolerance);

  expect_matrix_close<3, 3>(input.transpose(),
                            Matrix3x3{1, 4, 7, 2, 5, 8, 3, 6, 9}, kTolerance);
}

class VectorTest : public testing::Test {
public:
  const Vector2 left{1, 2};
  const Vector2 right{3, 4};
};

TEST_F(VectorTest, BasicOperations) {
  EXPECT_DOUBLE_EQ((left[1, 0]), 2);

  const auto sum = left + right;
  const auto diff = right - left;
  const auto quotient = right / left;

  expect_vector_close<2>(sum, Vector2{4, 6}, kTolerance);
  expect_vector_close<2>(diff, Vector2{2, 2}, kTolerance);
  expect_vector_close<2>(quotient, Vector2{3, 2}, kTolerance);

  auto inplace = left;
  inplace += right;
  expect_vector_close<2>(inplace, Vector2{4, 6}, kTolerance);
  inplace -= right;
  expect_vector_close<2>(inplace, left, kTolerance);
  inplace *= 2;
  expect_vector_close<2>(inplace, Vector2{2, 4}, kTolerance);
  inplace /= 2;
  expect_vector_close<2>(inplace, left, kTolerance);

  expect_vector_close<2>(left * 2, Vector2{2, 4}, kTolerance);
  expect_vector_close<2>(right / 2, Vector2{1.5, 2}, kTolerance);
  expect_vector_close<2>(left.hadamard(right), Vector2{3, 8}, kTolerance);
  EXPECT_DOUBLE_EQ(left.dot(right), 11);
  EXPECT_DOUBLE_EQ(left.sum(), 3);

  const auto incremented =
      left.elementwise([](const double value) { return value + 1; });
  expect_vector_close<2>(incremented, Vector2{2, 3}, kTolerance);
}

class TensorAndArrayTest : public testing::Test {
public:
  const Tensor tensor{{Matrix2x2{1, 2, 3, 4}, Matrix2x2{5, 6, 7, 8}}};
  const Tensor other{{Matrix2x2{1, 1, 1, 1}, Matrix2x2{2, 2, 2, 2}}};
};

TEST_F(TensorAndArrayTest, BasicArrayOperations) {
  EXPECT_EQ(tensor.size(), 2U);
  expect_matrix_close<2, 2>(tensor[0], Matrix2x2{1, 2, 3, 4}, kTolerance);
  expect_matrix_close<2, 2>(tensor[1], Matrix2x2{5, 6, 7, 8}, kTolerance);

  const auto sum = tensor + other;
  const auto diff = tensor - other;
  const auto quotient = tensor / other;
  const auto scaled = tensor * 2.0;

  expect_tensor_close<2, 2, 2>(
      sum, Tensor{Matrix2x2{2, 3, 4, 5}, Matrix2x2{7, 8, 9, 10}}, kTolerance);
  expect_tensor_close<2, 2, 2>(
      diff, Tensor{Matrix2x2{0, 1, 2, 3}, Matrix2x2{3, 4, 5, 6}}, kTolerance);
  expect_tensor_close<2, 2, 2>(
      quotient, Tensor{Matrix2x2{1, 2, 3, 4}, Matrix2x2{2.5, 3, 3.5, 4}},
      kTolerance);
  expect_tensor_close<2, 2, 2>(
      scaled, Tensor{Matrix2x2{2, 4, 6, 8}, Matrix2x2{10, 12, 14, 16}},
      kTolerance);

  auto inplace = tensor;
  inplace += other;
  expect_tensor_close<2, 2, 2>(
      inplace, Tensor{Matrix2x2{2, 3, 4, 5}, Matrix2x2{7, 8, 9, 10}},
      kTolerance);
  inplace -= other;
  expect_tensor_close<2, 2, 2>(inplace, tensor, kTolerance);
  inplace *= 2;
  expect_tensor_close<2, 2, 2>(
      inplace, Tensor{Matrix2x2{2, 4, 6, 8}, Matrix2x2{10, 12, 14, 16}},
      kTolerance);
  inplace /= 2;
  expect_tensor_close<2, 2, 2>(inplace, tensor, kTolerance);
}

TEST_F(TensorAndArrayTest, ReductionsAndTransformations) {
  EXPECT_DOUBLE_EQ(tensor.sum(), 36);
  EXPECT_DOUBLE_EQ(tensor.dot(other), 62);

  const auto hadamard = tensor.hadamard(other);
  expect_tensor_close<2, 2, 2>(
      hadamard, Tensor{Matrix2x2{1, 2, 3, 4}, Matrix2x2{10, 12, 14, 16}},
      kTolerance);

  const auto shifted = tensor + 1.0;
  expect_tensor_close<2, 2, 2>(
      shifted, Tensor{Matrix2x2{2, 3, 4, 5}, Matrix2x2{6, 7, 8, 9}},
      kTolerance);
}

TEST_F(TensorAndArrayTest, JsonRoundTrip) {
  nlohmann::json json = tensor;
  const auto restored = json.get<Tensor>();

  expect_tensor_close<2, 2, 2>(restored, tensor, kTolerance);
}

class TupleTest : public testing::Test {
public:
  const PairTuple left{Matrix2x2{1, 2, 3, 4}, Matrix2x2{5, 6, 7, 8}};
  const PairTuple right{Matrix2x2{2, 2, 2, 2}, Matrix2x2{1, 1, 1, 1}};
};

TEST_F(TupleTest, ConstructionAndAccess) {
  EXPECT_EQ(left.size(), 2U);
  expect_matrix_close<2, 2>(left.get<0>(), Matrix2x2{1, 2, 3, 4}, kTolerance);
  expect_matrix_close<2, 2>(left.get<1>(), Matrix2x2{5, 6, 7, 8}, kTolerance);
}

TEST_F(TupleTest, ArithmeticAndReductions) {
  const auto sum = left + right;
  const auto diff = right - left;
  const auto quotient = right / left;

  expect_matrix_close<2, 2>(sum.get<0>(), Matrix2x2{3, 4, 5, 6}, kTolerance);
  expect_matrix_close<2, 2>(sum.get<1>(), Matrix2x2{6, 7, 8, 9}, kTolerance);
  expect_matrix_close<2, 2>(diff.get<0>(), Matrix2x2{1, 0, -1, -2}, kTolerance);
  expect_matrix_close<2, 2>(diff.get<1>(), Matrix2x2{-4, -5, -6, -7},
                            kTolerance);
  expect_matrix_close<2, 2>(quotient.get<0>(), Matrix2x2{2, 1, 2.0 / 3.0, 0.5},
                            kTolerance);
  expect_matrix_close<2, 2>(quotient.get<1>(),
                            Matrix2x2{0.2, 1.0 / 6.0, 1.0 / 7.0, 0.125},
                            kTolerance);

  auto inplace = left;
  inplace += right;
  expect_matrix_close<2, 2>(inplace.get<0>(), Matrix2x2{3, 4, 5, 6},
                            kTolerance);
  expect_matrix_close<2, 2>(inplace.get<1>(), Matrix2x2{6, 7, 8, 9},
                            kTolerance);
  inplace -= right;
  expect_matrix_close<2, 2>(inplace.get<0>(), left.get<0>(), kTolerance);
  expect_matrix_close<2, 2>(inplace.get<1>(), left.get<1>(), kTolerance);
  inplace *= 2;
  expect_matrix_close<2, 2>(inplace.get<0>(), Matrix2x2{2, 4, 6, 8},
                            kTolerance);
  expect_matrix_close<2, 2>(inplace.get<1>(), Matrix2x2{10, 12, 14, 16},
                            kTolerance);
  inplace /= 2;
  expect_matrix_close<2, 2>(inplace.get<0>(), left.get<0>(), kTolerance);
  expect_matrix_close<2, 2>(inplace.get<1>(), left.get<1>(), kTolerance);

  EXPECT_DOUBLE_EQ(left.dot(right), 46);
  EXPECT_DOUBLE_EQ(left.sum(), 36);

  const auto hadamard = left.hadamard(right);
  expect_matrix_close<2, 2>(hadamard.get<0>(), Matrix2x2{2, 4, 6, 8},
                            kTolerance);
  expect_matrix_close<2, 2>(hadamard.get<1>(), Matrix2x2{5, 6, 7, 8},
                            kTolerance);

  const auto doubled =
      left.elementwise([](const auto &value) { return value + value; });
  expect_matrix_close<2, 2>(doubled.get<0>(), Matrix2x2{2, 4, 6, 8},
                            kTolerance);
  expect_matrix_close<2, 2>(doubled.get<1>(), Matrix2x2{10, 12, 14, 16},
                            kTolerance);
}

TEST_F(TupleTest, JsonRoundTrip) {
  nlohmann::json json = left;
  const auto restored = json.get<PairTuple>();

  expect_matrix_close<2, 2>(restored.get<0>(), left.get<0>(), kTolerance);
  expect_matrix_close<2, 2>(restored.get<1>(), left.get<1>(), kTolerance);
}

class SpanTest : public testing::Test {
public:
  Tensor tensor{{Matrix2x2{1, 2, 3, 4}, Matrix2x2{5, 6, 7, 8}}};
};

TEST_F(SpanTest, ConstructionAndIndexing) {
  StaticSpan<2, Matrix2x2> span(tensor);

  EXPECT_EQ(span.size(), 2U);
  expect_matrix_close<2, 2>(span[0], Matrix2x2{1, 2, 3, 4}, kTolerance);
  expect_matrix_close<2, 2>(span[1], Matrix2x2{5, 6, 7, 8}, kTolerance);
  EXPECT_EQ(span.begin(), tensor.get_data());
  EXPECT_EQ(span.end(), tensor.get_data() + 2);

  span[1][0, 0] = 42;
  EXPECT_DOUBLE_EQ((tensor[1][0, 0]), 42);
}

TEST_F(SpanTest, SubspanAndPointerConstruction) {
  StaticSpan<2, Matrix2x2> span(tensor);
  const auto subspan = span.template subspan<1, 2>();
  StaticSpan<1, Matrix2x2> pointer_span(tensor.get_data() + 1);

  EXPECT_EQ(subspan.size(), 1U);
  expect_matrix_close<2, 2>(subspan[0], Matrix2x2{5, 6, 7, 8}, kTolerance);
  expect_matrix_close<2, 2>(pointer_span[0], Matrix2x2{5, 6, 7, 8}, kTolerance);
  EXPECT_EQ(subspan.data(), tensor.get_data() + 1);
}
