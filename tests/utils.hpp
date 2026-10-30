#ifndef ML_UTILS_H
#define ML_UTILS_H
#include "../src/matrix/3DTensor.hpp"
#include "../src/matrix/Matrix.hpp"

#include "gtest/gtest.h"

template <size_t R, size_t C>
void expect_matrix_close(const Matrix<R, C> &actual,
                         const Matrix<R, C> &expected, const double tolerance) {
  for (size_t row = 0; row < R; ++row) {
    for (size_t col = 0; col < C; ++col) {
      EXPECT_NEAR((actual[row, col]), (expected[row, col]), tolerance)
          << "at [" << row << ", " << col << "]";
    }
  }
}

template <size_t Ch, size_t R, size_t C>
void expect_tensor_close(const Tensor3D<Ch, R, C> &actual,
                         const Tensor3D<Ch, R, C> &expected,
                         const double tolerance) {
  for (size_t ch = 0; ch < Ch; ++ch) {
    expect_matrix_close<R, C>(actual[ch], expected[ch], tolerance);
  }
}

template <size_t R>
void expect_vector_close(const Vector<R> &actual, const Vector<R> &expected,
                         const double tolerance) {
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR((actual[i]), (expected[i]), tolerance) << "at [" << i << "]";
  }
}

#endif // ML_UTILS_H