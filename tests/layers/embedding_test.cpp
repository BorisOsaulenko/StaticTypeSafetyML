#define protected public
#include "../../src/layers/Embedding.hpp"
#undef protected

#include "../utils.hpp"
#include <gtest/gtest.h>

constexpr double kTolerance = 0.005;
#define EMB_DIM 6
#define TOKENS 5

class EmbeddingLayerTest : public testing::Test {
protected:
  using Input = StaticArray<4, size_t>;
  using Output = Matrix<4, EMB_DIM>;
  using Params = Matrix<TOKENS, EMB_DIM>;

  Params params = {
      {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}, {2, 2, 2, 2, 2, 2},
      {3, 3, 3, 3, 3, 3}, {4, 4, 4, 4, 4, 4},
  };

  const Input input = {1, 4, 2, 1};

  const Output output = {
      {1, 1, 1, 1, 1, 1},
      {4, 4, 4, 4, 4, 4},
      {2, 2, 2, 2, 2, 2},
      {1, 1, 1, 1, 1, 1},
  };

  const Output delta = {
      {0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
      {0.2, 0.2, 0.2, 0.2, 0.2, 0.2},
      {-0.3, -0.3, -0.3, -0.3, -0.3, -0.3},
      {-0.2, -0.2, -0.2, -0.2, -0.2, -0.2},
  };

  Embedding<TOKENS, EMB_DIM> layer = {params};
};

TEST_F(EmbeddingLayerTest, ComputesZ) {
  const auto out = layer.compute_z<4>(input);
  expect_matrix_close(out, output, kTolerance);
}

TEST_F(EmbeddingLayerTest, ComputesZCached) {
  const auto cached = layer.compute_z_cached<4>(input);
  expect_matrix_close(cached.output, output, kTolerance);

  for (size_t i = 0; i < 3; i++) {
    EXPECT_EQ(input[i], cached.cache.a_prev[i]);
  }
}

TEST_F(EmbeddingLayerTest, ComputesDelta) {
  const auto cache = layer.compute_z_cached<4>(input);
  const auto new_delta = layer.compute_delta(delta, cache.cache);
  const Params expected_gradient = {{0, 0, 0, 0, 0, 0},
                                    {-0.1, -0.1, -0.1, -0.1, -0.1, -0.1},
                                    {-0.3, -0.3, -0.3, -0.3, -0.3, -0.3},
                                    {0, 0, 0, 0, 0, 0},
                                    {0.2, 0.2, 0.2, 0.2, 0.2, 0.2}};

  expect_matrix_close(new_delta.gradient, expected_gradient, kTolerance);
}

#undef EMB_DIM
#undef TOKENS