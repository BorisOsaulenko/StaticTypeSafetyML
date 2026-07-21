// //
// // Created by Boris Osaulenko on 16/07/2026.
// //
//
// #include "../../src/layers/Flatten.h"
// #include "../../src/matrix/3DTensor.h"
// #include "../../src/matrix/Matrix.h"
// #include "utils.h"
//
// #include <gtest/gtest.h>
//
// static constexpr double kTolerance = 0.05;
//
// class FlattenTest : public testing::Test {
// public:
//   static const inline Tensor3D<2, 2, 2> input = {Matrix<2, 2>{1, 2, 3, 4},
//                                                  Matrix<2, 2>{5, 6, 7, 8}};
//
//   static const inline Vector<8> delta = {1, 2, 3, 4, -1, -2, -3, -4};
//
// protected:
//   Flatten<2, 2, 2> flatten_layer;
//
//   void SetUp() override {}
// };
//
// TEST_F(FlattenTest, ForwardPass) {
//   const auto output = flatten_layer.forward(input);
//   const Vector<8> expected = {1, 2, 3, 4, 5, 6, 7, 8};
//
//   expect_vector_close(output, expected, kTolerance);
// }
//
// TEST_F(FlattenTest, BackwardPass) {
//   flatten_layer.forward(input);
//   const auto new_delta = flatten_layer.backward(delta);
//   const Tensor3D<2, 2, 2> expected = {Matrix<2, 2>{1, 2, 3, 4},
//                                       Matrix<2, 2>{-1, -2, -3, -4}};
//
//   expect_tensor_close(new_delta, expected, kTolerance);
// }