// #include "../../src/Network.h"
// #include "../../src/layers/Dense.h"
// #include "../../src/loss/Loss.h"
// #include "../../src/lr_optimizer/Adam.h"
// #include "utils.h"
//
// #include <gtest/gtest.h>
//
// static constexpr double kTolerance = 0.05;
//
// TEST(LinearRegression, LearnsExpectedWeightsAndBias) {
//   Matrix<1, 3>::set_seed(42);
//   Vector<1>::set_seed(42);
//
//   const AdamFactory optimizer_factory(0.04);
//   auto layer =
//       DenseLayer<3, 1>(ActivationFunctionEnum::Identity, optimizer_factory);
//
//   const std::vector inputs = {
//       Vector<3>({1.0, 0.0, 0.0}),
//       Vector<3>({0.0, 1.0, 0.0}),
//       Vector<3>({0.0, 0.0, 1.0}),
//       Vector<3>({1.0, 1.0, 1.0}),
//   };
//
//   const std::vector targets = {
//       Vector<1>({7.0}),
//       Vector<1>({6.0}),
//       Vector<1>({5.5}),
//       Vector<1>({8.5}),
//   };
//
//   Network network(LossEnum::MSE, inputs, targets, layer);
//
//   network.train(nullptr, 1000, 4);
//   const auto expected_weights = Matrix<1, 3>({2.0, 1.0, 0.5});
//   const auto expected_bias_vector = Vector<1>({5.0});
//
//   expect_matrix_close(layer.get_weights(), expected_weights, kTolerance);
//   expect_vector_close(layer.get_bias(), expected_bias_vector, kTolerance);
// }
