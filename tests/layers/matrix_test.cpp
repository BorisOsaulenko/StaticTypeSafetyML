// //
// // Created by Boris Osaulenko on 16/07/2026.
// //
//
// #include "../../src/matrix/Matrix.h"
// #include "utils.h"
//
// #include <gtest/gtest.h>
//
// static constexpr double kTolerance = 0.05;
//
// class MatrixTest : public testing::Test {
// public:
//   const Matrix<3, 3> input{1, 2, 3, 4, 5, 6, 7, 8, 9};
//   const Matrix<2, 2> kernel{1, 2, 3, 4};
//
// protected:
//   void SetUp() override {}
// };
//
// TEST_F(MatrixTest, CorrectCrossCorrelation) {
//   const auto expectedValidCrossCorrelation = Matrix<2, 2>{37, 47, 67, 77};
//   expect_matrix_close(input.valid_cross_correlation(kernel),
//                       expectedValidCrossCorrelation, kTolerance);
// }
//
// TEST_F(MatrixTest, CorrectConvolution) {
//   const auto expectedFullConvolution =
//       Matrix<4, 4>{1, 4, 7, 6, 7, 23, 33, 24, 19, 53, 63, 42, 21, 52, 59,
//       36};
//   expect_matrix_close(input.full_convolution(kernel),
//   expectedFullConvolution,
//                       kTolerance);
// }