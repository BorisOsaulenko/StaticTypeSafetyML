#define protected public
#include "../../src/layers/AddAndNorm.h"
#undef protected

#include "../utils.hpp"
#include <gtest/gtest.h>

constexpr double kTolerance = 0.005;

class AddAndNormTest : public testing::Test {
public:
  static constexpr size_t Emb = 4;
  using Input = Matrix<3, Emb>;
  using Output = Input;
  using Param = Matrix<2, Emb>;

  // row 0: gamma (per embedding dim), row 1: beta (per embedding dim)
  Param params = {{1.5, 0.8, -1.2, 2.0}, {-0.2, 0.1, 0.3, 0.0}};
  AddAndNorm<Emb> layer = {params};

  // already x + Sublayer(x)
  Input input = {
      {1.0, -0.5, 0.8, 0.4}, {0.5, 2.0, -1.2, 0.9}, {-1.0, 0.3, 1.5, -0.7}};

  // per-row stats (eps = 1e-5)
  // means     = {0.425, 0.55, 0.025}
  // variances = {0.331875, 1.3225, 0.956875}
  const Vector<3> expected_means = {0.425, 0.55, 0.025};
  const Vector<3> expected_inv_std = {1.7358259948, 0.8695619298, 1.0222806176};

  // x_hat = (x - mean) * inv_std
  // {{0.998099947, -1.6056390452, 0.6509347481, -0.0433956499},
  //  {-0.0434780965, 1.2608647983, -1.5217333772, 0.3043466754},
  //  {-1.047837633, 0.2811271698, 1.5078639109, -0.7411534477}}

  // y = gamma[c] * x_hat + beta[c]
  Output output = {{1.2971499205, -1.1845112362, -0.4811216977, -0.0867912997},
                   {-0.2652171447, 1.1086918386, 2.1260800526, 0.6086933509},
                   {-1.7717564495, 0.3249017359, -1.5094366931, -1.4823068955}};

  Output delta = {
      {0.3, 0.5, -0.2, 0.8}, {-0.7, 0.1, 0.6, -0.3}, {0.2, -0.4, 0.9, 0.1}};

  // row 0: dgamma[c] = sum_r delta * x_hat, row 1: dbeta[c] = sum_r delta
  const Param expected_gradient = {
      {0.120297125, -0.7891839107, 0.3138505439, -0.2001358673},
      {-0.2, 0.2, 1.3, 0.6}};

  Output expected_new_delta = {
      {-0.340170979, -0.5470935088, -0.7207119352, 1.607976423},
      {-0.405201141, 0.2769634239, 0.2222534338, -0.0940157168},
      {-0.0473753674, 0.0595857708, -0.0335550791, 0.0213446757}};
};

TEST_F(AddAndNormTest, ComputesZ) {
  const Output z = layer.compute_z(input);
  expect_matrix_close(z, output, kTolerance);
}

TEST_F(AddAndNormTest, ComputesZCached) {
  const auto [z, cache] = layer.compute_z_cached(input);

  expect_matrix_close(z, output, kTolerance);
  expect_matrix_close(cache.a_prev, input, kTolerance);
  expect_vector_close(cache.means, expected_means, kTolerance);
  expect_vector_close(cache.variances_inv, expected_inv_std, kTolerance);
}

TEST_F(AddAndNormTest, ComputeDelta) {
  const auto [z, cache] = layer.compute_z_cached(input);
  const auto [actual_delta, gradient] = layer.compute_delta(delta, cache);

  expect_matrix_close(actual_delta, expected_new_delta, kTolerance);
  expect_matrix_close(gradient, expected_gradient, kTolerance);
}