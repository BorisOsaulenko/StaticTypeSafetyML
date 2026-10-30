#include "../../src/layers/Flatten.hpp"
#include "../../src/matrix/3DTensor.hpp"
#include "../../src/matrix/Matrix.hpp"
#include "../utils.hpp"

#include <gtest/gtest.h>

static constexpr double kTolerance = 0.05;

using Input = Tensor3D<2, 2, 2>;
using Output = Vector<8>;

class FlattenTest : public testing::Test {
public:
  static const inline Input input_elem = {Matrix<2, 2>{1, 2, 3, 4},
                                          Matrix<2, 2>{5, 6, 7, 8}};
  static const inline StaticArray<1, Input> input = {input_elem};

  static const inline Output delta_elem = {1, 2, 3, 4, -1, -2, -3, -4};
  static const inline StaticArray<1, Output> delta = {delta_elem};

protected:
  Flatten flatten_layer;

  void SetUp() override {}
};

TEST_F(FlattenTest, ForwardPass) {
  const Output output = flatten_layer.compute_z(input_elem);
  const Output expected = {1, 2, 3, 4, 5, 6, 7, 8};

  expect_vector_close(output, expected, kTolerance);
}

TEST_F(FlattenTest, BackwardPass) {
  StaticArray<1, LayerCache<Flatten, Input>> cache;
  flatten_layer.forward_vectorized(input, cache);
  const StaticArray<1, Input> new_delta =
      flatten_layer.backward_vectorized<1, Input>(delta, cache);
  const Tensor3D<2, 2, 2> expected = {Matrix<2, 2>{1, 2, 3, 4},
                                      Matrix<2, 2>{-1, -2, -3, -4}};

  expect_tensor_close(new_delta[0], expected, kTolerance);
}