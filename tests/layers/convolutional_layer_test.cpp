// //
// // Created by Boris Osaulenko on 15/07/2026.
// //
//
// #include "../../src/Network.h"
// #include "../../src/layers/Convolutional.h"
// #include "utils.h"
// #include <gtest/gtest.h>
//
// constexpr double kTolerance = 0.05;
//
// class ConvolutionalTest : public testing::Test {
// public:
//   using Kernel = Tensor3D<2, 2, 2>;
//   using KernelArray = AlgebraicArray<2, Kernel>;
//   using json = nlohmann::json;
//
//   static const inline Tensor3D<2, 3, 3> input = {
//       Matrix<3, 3>{1, 2, 3, 4, 5, 6, 7, 8, 9},
//       Matrix<3, 3>{9, 8, 7, 6, 5, 4, 3, 2, 1}};
//
//   static const inline Tensor3D<2, 2, 2> delta = {Matrix<2, 2>{1, 2, 3, 4},
//                                                  Matrix<2, 2>{-1, 0, 2, 1}};
//
// protected:
//   using Conv = Convolutional<2, 3, 3, 2, 2, 2>;
//   AdamFactory optimizer_factory;
//
//   static const inline Kernel kernel1 = {Matrix<2, 2>{1, 0, 0, 1},
//                                         Matrix<2, 2>{0, 1, 1, 0}};
//   static const inline Kernel kernel2 = {Matrix<2, 2>{1, 1, 1, 1},
//                                         Matrix<2, 2>{-1, -1, -1, -1}};
//   std::vector<Kernel> kernels{kernel1, kernel2};
//
//   Conv convolutional_layer{Conv::Params{KernelArray(kernels), bias},
//                            ActivationFunctionEnum::Identity,
//                            optimizer_factory};
//   static const inline Vector<2> bias = {1, -1};
//   void SetUp() override {}
// };
//
// TEST_F(ConvolutionalTest, ForwardPass) {
//   const auto expected_output1 = Matrix<2, 2>{21, 21, 21, 21};
//   const auto expected_output2 = Matrix<2, 2>{-17, -9, 7, 15};
//
//   const auto output = convolutional_layer.forward(input);
//
//   expect_matrix_close(output[0], expected_output1, kTolerance);
//   expect_matrix_close(output[1], expected_output2, kTolerance);
// }
//
// TEST_F(ConvolutionalTest, BackwardPass_CorrectKernelGradients) {
//   convolutional_layer.forward(input);
//
//   const Tensor3D<2, 2, 2> expected_kernel_gradients1 = {
//       Matrix<2, 2>{37, 47, 67, 77}, Matrix<2, 2>{63, 53, 33, 23}};
//   const Tensor3D<2, 2, 2> expected_kernel_gradients2 = {
//       Matrix<2, 2>{12, 14, 18, 20}, Matrix<2, 2>{8, 6, 2, 0}};
//   const auto expected_bias_gradients = Vector<2>{1, 1};
//
//   convolutional_layer.backward(delta);
//
//   const auto kernel_gradients = convolutional_layer.get_kernel_gradients();
//   const auto bias_gradients = convolutional_layer.get_bias_gradients();
//
//   expect_matrix_close(kernel_gradients[0][0], expected_kernel_gradients1[0],
//                       kTolerance);
//   expect_matrix_close(kernel_gradients[0][1], expected_kernel_gradients1[1],
//                       kTolerance);
//   expect_matrix_close(kernel_gradients[1][0], expected_kernel_gradients2[0],
//                       kTolerance);
//   expect_matrix_close(kernel_gradients[1][1], expected_kernel_gradients2[1],
//                       kTolerance);
// }
//
// TEST_F(ConvolutionalTest, BackwardPass_CorrectBiasGradients) {
//   convolutional_layer.forward(input);
//   convolutional_layer.backward(delta);
//
//   const auto expected_bias_gradients = Vector<2>{10, 2};
//   const auto bias_gradients = convolutional_layer.get_bias_gradients();
//
//   expect_vector_close(bias_gradients, expected_bias_gradients, kTolerance);
// }
//
// TEST_F(ConvolutionalTest, BackwardPass_CorrectReturnDelta) {
//   convolutional_layer.forward(input);
//   const auto new_delta = convolutional_layer.backward(delta);
//
//   const Tensor3D<2, 3, 3> expected_delta_prev = {
//       Matrix<3, 3>{0, 1, 0, 4, 7, 3, 2, 6, 5},
//       Matrix<3, 3>{1, 2, 2, 0, 3, 3, 1, 1, -1}};
//
//   expect_tensor_close(new_delta, expected_delta_prev, kTolerance);
// }
//
// TEST(ConvolutionalLearning, LearnsExpectedWeights) {
//   // Similar to the linear regression test, but for convolutional layers.
//   // We create a simple relationship: output = sum(kernel * input_patch) +
//   bias Matrix<3, 3>::set_seed(42); Vector<1>::set_seed(42);
//
//   AdamFactory optimizer_factory(0.1);
//   auto conv_layer = Convolutional<1, 3, 3, 2, 2, 1>(
//       ActivationFunctionEnum::Identity, optimizer_factory);
//
//   // Simple inputs: each has only one non-zero element at specific positions
//   const std::vector inputs = {
//       Tensor3D<1, 3, 3>{Matrix<3, 3>{1, 0, 0, 0, 0, 0, 0, 0, 0}},
//       Tensor3D<1, 3, 3>{Matrix<3, 3>{0, 2, 0, 0, 0, 0, 0, 0, 0}},
//       Tensor3D<1, 3, 3>{Matrix<3, 3>{0, 0, 0, 3, 0, 0, 0, 0, 0}},
//       Tensor3D<1, 3, 3>{Matrix<3, 3>{0, 0, 0, 0, 4, 0, 0, 0, 0}},
//   };
//
//   // Targets are designed so the network learns a simple kernel:
//   // Expected: kernel = [[1.0, 1.0], [1.0, 1.0]], bias = 0.0
//   // Then output = sum of the 4 input values at kernel positions
//   const std::vector targets = {
//       Tensor3D<1, 2, 2>{Matrix<2, 2>{1.0, 0, 0, 0}},
//       Tensor3D<1, 2, 2>{Matrix<2, 2>{2.0, 2.0, 0, 0}},
//       Tensor3D<1, 2, 2>{Matrix<2, 2>{3.0, 0, 3.0, 0}},
//       Tensor3D<1, 2, 2>{Matrix<2, 2>{4.0, 4.0, 4.0, 4.0}},
//   };
//
//   Network network(LossEnum::MSE, inputs, targets, conv_layer);
//
//   // Train for 1000 epochs with batch size 1
//   network.train(nullptr, 1000, 1);
//
//   // Get learned kernel and bias
//   const auto learned_kernels = conv_layer.get_kernels();
//   const auto learned_bias = conv_layer.get_bias();
//
//   // After training, the network should have learned kernel values around 0.5
//   // and a bias around 1.6 based on the learning dynamics
//   const Matrix<2, 2> expected_kernel = {1, 1, 1, 1};
//   const Vector<1> expected_bias = {0};
//
//   // Assert learned parameters are close to expected values with reasonable
//   // tolerance
//   expect_matrix_close(learned_kernels[0][0], expected_kernel, kTolerance);
//   expect_vector_close(learned_bias, expected_bias, kTolerance);
// }