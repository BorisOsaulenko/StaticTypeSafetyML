//
// Created by Boris Osaulenko on 13/07/2026.
//

#ifndef ML_MAXPOOL2X2_H
#define ML_MAXPOOL2X2_H
#include "../matrix/3DTensor.h"
#include "NoParamsLayer.h"
#include <nlohmann/json.hpp>

template <int Ch_in, int R_in, int C_in, std::size_t batch_size>
class MaxPool2x2 final
    : public NoParamsLayer<Tensor3D<Ch_in, R_in, C_in>,
                           Tensor3D<Ch_in, R_in / 2, C_in / 2>, batch_size> {
  static_assert(R_in % 2 == 0, "Input rows must be even for MaxPool2x2.");
  static_assert(C_in % 2 == 0, "Input columns must be even for MaxPool2x2.");

  using json = nlohmann::json;

public:
  using Input = Tensor3D<Ch_in, R_in, C_in>;
  using Output = Tensor3D<Ch_in, R_in / 2, C_in / 2>;

  MaxPool2x2() : NoParamsLayer<Input, Output, batch_size>() {}

protected:
  Output compute_z(const Input &inp) const override {
    Output result{};
    for (std::size_t ch = 0; ch < Ch_in; ++ch) {
      for (std::size_t r = 0; r < R_in / 2; ++r) {
        for (std::size_t c = 0; c < C_in / 2; ++c) {
          result[ch][r, c] = inp[ch][r * 2, c * 2];
          for (std::size_t dr = 0; dr < 2; ++dr) {
            for (std::size_t dc = 0; dc < 2; ++dc) {
              if (inp[ch][r * 2 + dr, c * 2 + dc] > result[ch][r, c]) {
                result[ch][r, c] = inp[ch][r * 2 + dr, c * 2 + dc];
              }
            }
          }
        }
      }
    }
    return result;
  }

  Input compute_delta(const Output &dz,
                      const Input &a_previous) const override {
    Input new_delta;
    for (std::size_t ch = 0; ch < Ch_in; ++ch) {
      for (std::size_t r = 0; r < R_in / 2; ++r) {
        for (std::size_t c = 0; c < C_in / 2; ++c) {
          max_in_batch(new_delta, a_previous, ch, 2 * r, 2 * c) = dz[ch][r, c];
        }
      }
    }
    return new_delta;
  }

private:
  static double &max_in_batch(Input &delta, Input a_previous,
                              const std::size_t &ch, const std::size_t &r,
                              const std::size_t &c) {
    double *max = &delta[ch][r, c];
    double max_val = a_previous[ch][r, c];
    if (a_previous[ch][r + 1, c] > max_val) {
      max = &delta[ch][r + 1, c];
      max_val = a_previous[ch][r + 1, c];
    }
    if (a_previous[ch][r, c + 1] > max_val) {
      max = &delta[ch][r, c + 1];
      max_val = a_previous[ch][r, c + 1];
    }
    if (a_previous[ch][r + 1, c + 1] > max_val)
      max = &delta[ch][r + 1, c + 1];

    return *max;
  }
};

#endif // ML_MAXPOOL2X2_H
