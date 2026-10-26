#define protected public
#include "../../src/layers/Dense.h"
#undef protected
#include "../../src/lr_optimizer/Adam.h"

#include "../utils.h"
#include <gtest/gtest.h>

namespace {

constexpr double kTolerance = 0.005;

using Input = Vector<3>;
using Output = Vector<2>;
using Layer = DenseLayer<3, 2>;
using Gradient = DenseLayer<3, 2>::Params;

const StaticArray<2, Input> inputs = {
    Input{1.0, 2.0, -1.0},
    Input{-2.0, 0.5, 3.0},
};

const StaticArray<2, Output> delta = {
    Output{2.0, -1.0},
    Output{-0.5, 3.0},
};

Layer make_layer() {
  const Gradient params{
      Matrix<2, 3>{1.0, -1.0, 0.5, 0.5, 2.0, -0.5},
      Vector<2>{0.1, -0.2},
  };
  const AdamFactory optimizer_factory;
  return Layer{params, ActivationFunctionEnum::Sigmoid};
}

} // namespace

TEST(DenseLayerTest, ForwardPassWithSigmoidActivation) {
  auto layer = make_layer();
  StaticArray<2, LayerCache<Layer, Input>> z_cache;

  const StaticArray<2, Output> output =
      layer.forward_vectorized<2>(StaticSpan<2, const Input>(inputs), z_cache);

  const StaticArray<2, Output> expected = {
      Output{0.1978161114, 0.9918374288},
      Output{0.2890504974, 0.1544652651},
  };

  expect_vector_close<2>(output[0], expected[0], kTolerance);
  expect_vector_close<2>(output[1], expected[1], kTolerance);
}

TEST(DenseLayerTest, BackwardPassCalculatesGradient) {
  auto layer = make_layer();
  StaticArray<2, LayerCache<Layer, Input>> cache;
  layer.forward_vectorized<2>(inputs, cache);

  layer.backward_vectorized<2, Input>(delta, cache);

  const auto &weights = layer.get_weights_gradients();
  const Matrix<2, 3> expected_weights_gradient{
      0.5228700994,  0.5832394864, -0.6256202523,
      -0.7917304297, 0.1797167334, 1.1835476695,
  };

  const auto &bias = layer.get_bias_gradients();
  const Vector<2> expected_bias_gradient{
      0.2146196396,
      0.3837213026,
  };

  expect_matrix_close<2, 3>(weights, expected_weights_gradient, kTolerance);
  expect_vector_close<2>(bias, expected_bias_gradient, kTolerance);
}

TEST(DenseLayerTest, BackwardPassReturnsNewDelta) {
  auto layer = make_layer();
  StaticArray<2, LayerCache<Layer, Input>> cache;
  layer.forward_vectorized<2>(inputs, cache);

  const auto new_delta = layer.backward_vectorized<2, Input>(delta, cache);

  const StaticArray<2, Input> expected_previous_delta{
      Input{0.3133218302, -0.3335616749, 0.1627328685},
      Input{0.0931584701, 0.8863846318, -0.2472836956},
  };

  expect_vector_close<3>(new_delta[0], expected_previous_delta[0], kTolerance);
  expect_vector_close<3>(new_delta[1], expected_previous_delta[1], kTolerance);
}
