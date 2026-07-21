//
// Created by Boris Osaulenko on 13/07/2026.
//

#ifndef ML_FLATTEN_H
#define ML_FLATTEN_H
#include "../matrix/3DTensor.h"
#include "NoParamsLayer.h"

#include <nlohmann/json.hpp>

template <int Ch_in, int R_in, int C_in, std::size_t batch_size>
class Flatten : public NoParamsLayer<Tensor3D<Ch_in, R_in, C_in>,
                                     Vector<Ch_in * R_in * C_in>, batch_size> {
  using json = nlohmann::json;

public:
  using Input = Tensor3D<Ch_in, R_in, C_in>;
  using Output = Vector<Ch_in * R_in * C_in>;

  Flatten() : NoParamsLayer<Input, Output, batch_size>() {}

protected:
  Output compute_z(const Input &inp) const override {
    Output result;
    for (std::size_t ch = 0; ch < Ch_in; ++ch) {
      for (std::size_t r = 0; r < R_in; ++r) {
        for (std::size_t c = 0; c < C_in; ++c) {
          result[ch * R_in * C_in + r * C_in + c] = inp[ch][r, c];
        }
      }
    }
    return result;
  }

  Input compute_delta(const Output &delta,
                      [[maybe_unused]] const Input &a_previous) const override {
    Input new_delta;
    for (std::size_t ch = 0; ch < Ch_in; ++ch) {
      for (std::size_t r = 0; r < R_in; ++r) {
        for (std::size_t c = 0; c < C_in; ++c) {
          new_delta[ch][r, c] = delta[ch * R_in * C_in + r * C_in + c];
        }
      }
    }
    return new_delta;
  }
};

#endif // ML_FLATTEN_H