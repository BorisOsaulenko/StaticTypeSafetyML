#ifndef ML_MAXPOOL2X2_H
#define ML_MAXPOOL2X2_H
#include "../matrix/3DTensor.h"
#include "LayerConcepts.h"
#include "NoParamsLayer.h"

#define DIM_TEMPLATED template <size_t Ch_in, size_t R_in, size_t C_in>
#define REQUIRES_EVEN_DIM requires(R_in % 2 == 0 && C_in % 2 == 0)
#define INPUT Tensor3D<Ch_in, R_in, C_in>
#define OUTPUT Tensor3D<Ch_in, R_in / 2, C_in / 2>
#define CACHE Cache<Ch_in, R_in, C_in>

class MaxPool2x2 final : public NoParamsLayer<MaxPool2x2> {
  DIM_TEMPLATED
  struct Cache {
    INPUT a_previous;
  };

public:
  MaxPool2x2() : NoParamsLayer<MaxPool2x2>() {}

  DIM_TEMPLATED
  OUTPUT compute_z(const INPUT &inp) const REQUIRES_EVEN_DIM {
    OUTPUT result{};
    for (size_t ch = 0; ch < Ch_in; ++ch) {
      for (size_t r = 0; r < R_in / 2; ++r) {
        for (size_t c = 0; c < C_in / 2; ++c) {
          result[ch][r, c] = inp[ch][r * 2, c * 2];
          for (size_t dr = 0; dr < 2; ++dr) {
            for (size_t dc = 0; dc < 2; ++dc) {
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

  DIM_TEMPLATED
  ForwardCachedResult<OUTPUT, CACHE>
  compute_z_cached(const INPUT &inp) const REQUIRES_EVEN_DIM {
    OUTPUT z = compute_z(inp);
    return {z, inp};
  }

  DIM_TEMPLATED
  INPUT compute_delta(const OUTPUT &delta, const CACHE &cache) const {
    INPUT new_delta;
    for (size_t ch = 0; ch < Ch_in; ++ch) {
      for (size_t r = 0; r < R_in / 2; ++r) {
        for (size_t c = 0; c < C_in / 2; ++c) {
          max_in_batch(new_delta, cache.a_previous, ch, 2 * r, 2 * c) =
              delta[ch][r, c];
        }
      }
    }
    return new_delta;
  }

private:
  DIM_TEMPLATED
  static double &max_in_batch(INPUT &delta, INPUT a_previous, const size_t &ch,
                              const size_t &r, const size_t &c) {
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

#undef DIM_TEMPLATED
#undef REQUIRES_EVEN_DIM
#undef INPUT
#undef OUTPUT
#undef CACHE

#endif // ML_MAXPOOL2X2_H
