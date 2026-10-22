#ifndef ML_3DTENSOR_H
#define ML_3DTENSOR_H

#include "Matrix.h"
#include "StaticArray.h"

#include <nlohmann/json.hpp>
#include <ostream>
#include <vector>

// Thin wrapper: Tensor3D is an AlgebraicArray of matrices.
template <size_t Ch, size_t R, size_t C>
using Tensor3D = StaticArray<Ch, Matrix<R, C>>;

// Stream operator for Tensor3D (prints each channel using Matrix's stream)
template <size_t Ch, size_t R, size_t C>
std::ostream &operator<<(std::ostream &os, const Tensor3D<Ch, R, C> &t) {
  for (size_t ch = 0; ch < Ch; ++ch) {
    os << "Channel " << ch << ":\n";
    os << t[ch];
  }
  return os;
}

// JSON helpers: serialize as a vector of matrices
template <size_t Ch, size_t R, size_t C>
void to_json(nlohmann::json &j, const Tensor3D<Ch, R, C> &t) {
  j = t.get_data();
}

template <size_t Ch, size_t R, size_t C>
void from_json(const nlohmann::json &j, Tensor3D<Ch, R, C> &t) {
  auto v = j.get<std::vector<Matrix<R, C>>>();
  t = Tensor3D<Ch, R, C>::from_vector(v);
}

#endif // ML_3DTENSOR_H
