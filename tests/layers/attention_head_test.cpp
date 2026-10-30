#define protected public
#include "../../src/layers/AttentionHead.h"
#undef protected

#include "../utils.h"
#include <gtest/gtest.h>

constexpr double kTolerance = 0.005;
#define EMB_DIM 6
#define TOKENS 5

class AddAndNormTest : public testing::Test {
protected:
  using Param = Matrix<4, 2>;
  using Input = Matrix<3, 4>;
  using Output = Input;

  Param Wq = {{0.2, 0.5}, {-0.1, 0.3}, {0.4, -0.2}, {0.3, -0.1}};
  Param Wk = {{-0.3, 0.2}, {0.6, -0.4}, {0.1, 0.5}, {0.2, -0.3}};
  Param Wd = {{0.1, -0.6}, {0.7, 0.2}, {-0.3, 0.4}, {0.4, 0.1}};
  Param Wu = {{0.5, 0.1}, {-0.2, 0.4}, {0.3, -0.5}, {-0.4, 0.6}};

  Input input = {
      {1.0, -0.5, 0.8, 0.4}, {0.5, 2.0, -1.2, 0.9}, {-1.0, 0.3, 1.5, -0.7}};
  Output output = {{-0.199, -0.07, 0.071, -0.072},
                   {-0.04003287824, -0.1308001741, 0.1628335962, -0.1948670184},
                   {0.5651012405, -0.2220300143, 0.3339564946, -0.4458829749}};

  Output delta = {
      {0.3, 0.5, -0.2, 0.8}, {-0.7, 0.1, 0.6, -0.3}, {0.2, -0.4, 0.9, 0.1}};
  Output expected_new_delta = {
      {-0.1777029421, -0.1956577471, 0.2251886428, -0.1172733457},
      {0.2198845157, 0.2015908798, -0.3273095783, 0.1086329919},
      {-0.1282543393, -0.2199423638, 0.1843476529, 0.09641988149}};

  Param dWq = {{-0.2181258132, 0.5757419993},
               {0.01317250039, -0.09685369782},
               {0.3381279568, -0.8794925365},
               {-0.1660582478, 0.4224277232}};
  Param dWk = {{0.04073776435, -0.1648330767},
               {0.08746126549, -0.3404349656},
               {-0.1123756555, 0.441974545},
               {0.05736715635, -0.2275866061}};
  Param dWd = {{-0.2809351267, 0.1841178743},
               {0.6638697889, -0.9397915723},
               {-0.4126415016, 0.5118801002},
               {0.03015272257, -0.1638381811}};
  Param dWu = {{0.1360086057, 0.1333896003},
               {-0.6176850566, -0.2070125575},
               {1.073542125, -0.1239967306},
               {-0.1471623185, -0.1709885323}};

  StaticArray<4, Param> params = {Wq, Wk, Wd, Wu};
  AttentionHead<4, 2> attention_head = {params,
                                        ActivationFunctionEnum::Identity};

  const Matrix<3, 2> expected_queries = {
      {0.69, 0.15}, {-0.31, 1.0}, {0.16, -0.64}};
  const Matrix<3, 2> expected_keys = {
      {-0.44, 0.68}, {1.11, -1.57}, {0.49, 0.64}};
  const Matrix<3, 3> expected_attention = {
      {1.0, 0.0, 0.0},
      {0.8733329707, 0.1266670293, 0.0},
      {0.1841563191, 0.6075176352, 0.2083260457}};
  const Matrix<3, 4> expected_values = {{-0.199, -0.07, 0.071, -0.072},
                                        {1.056, -0.55, 0.796, -1.042},
                                        {-0.191, 0.6, -0.781, 0.962}};
};

TEST_F(AddAndNormTest, ComputesZ) {
  const Output z = attention_head.compute_z(input);
  expect_matrix_close(z, output, kTolerance);
}

TEST_F(AddAndNormTest, ComputesZCached) {
  const auto [z, cache] = attention_head.compute_z_cached(input);

  expect_matrix_close(z, output, kTolerance);
  expect_matrix_close(cache.a_prev, input, kTolerance);
  expect_matrix_close(cache.queries, expected_queries, kTolerance);
  expect_matrix_close(cache.keys, expected_keys, kTolerance);
  expect_matrix_close(cache.attention_softmaxed, expected_attention,
                      kTolerance);
  expect_matrix_close(cache.values_matrix, expected_values, kTolerance);
}

TEST_F(AddAndNormTest, ComputeDelta) {
  const auto [z, cache] = attention_head.compute_z_cached(input);
  const auto [actual_delta, gradient] =
      attention_head.compute_delta(delta, cache);

  expect_matrix_close(actual_delta, expected_new_delta, kTolerance);
  expect_matrix_close(gradient[0], dWq, kTolerance);
  expect_matrix_close(gradient[1], dWk, kTolerance);
  expect_matrix_close(gradient[2], dWd, kTolerance);
  expect_matrix_close(gradient[3], dWu, kTolerance);
}