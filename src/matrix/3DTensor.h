//
// Created by Boris Osaulenko on 11/07/2026.
//

#ifndef ML_3DTENSOR_H
#define ML_3DTENSOR_H

#include "../layers/Dense.h"
#include "AlgebraicArray.h"
#include "Matrix.h"

#include <nlohmann/json.hpp>
#include <ostream>
#include <vector>

// Thin wrapper: Tensor3D is an AlgebraicArray of matrices.
template <int Ch, int R, int C>
using Tensor3D = AlgebraicArray<Ch, Matrix<R, C>>;

// Stream operator for Tensor3D (prints each channel using Matrix's stream)
template <int Ch, int R, int C>
std::ostream &operator<<(std::ostream &os, const Tensor3D<Ch, R, C> &t) {
  for (std::size_t ch = 0; ch < static_cast<std::size_t>(Ch); ++ch) {
    os << "Channel " << ch << ":\n";
    os << t[ch];
  }
  return os;
}

// JSON helpers: serialize as a vector of matrices
template <int Ch, int R, int C>
void to_json(nlohmann::json &j, const Tensor3D<Ch, R, C> &t) {
  j = t.get_data();
}

template <int Ch, int R, int C>
void from_json(const nlohmann::json &j, Tensor3D<Ch, R, C> &t) {
  auto v = j.get<std::vector<Matrix<R, C>>>();
  t = Tensor3D<Ch, R, C>::from_vector(v);
}

#endif // ML_3DTENSOR_H
