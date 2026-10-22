#ifndef ML_FLATTEN_H
#define ML_FLATTEN_H
#include "../matrix/3DTensor.h"
#include "NoParamsLayer.h"

#define DIM_TEMPLATED template <size_t Ch_in, size_t R_in, size_t C_in>
#define INPUT Tensor3D<Ch_in, R_in, C_in>
#define OUTPUT Vector<Ch_in * R_in * C_in>
#define CACHE Cache<Ch_in, R_in, C_in>

class Flatten : public NoParamsLayer<Flatten> {
  DIM_TEMPLATED
  struct Cache {};

public:
  Flatten() : NoParamsLayer<Flatten>() {}

  DIM_TEMPLATED
  OUTPUT compute_z(const INPUT &inp) const {
    OUTPUT result;
    for (size_t ch = 0; ch < Ch_in; ++ch) {
      for (size_t r = 0; r < R_in; ++r) {
        for (size_t c = 0; c < C_in; ++c) {
          result[ch * R_in * C_in + r * C_in + c] = inp[ch][r, c];
        }
      }
    }
    return result;
  }

  DIM_TEMPLATED
  ForwardCachedResult<OUTPUT, CACHE> compute_z_cached(const INPUT &inp) const {
    OUTPUT z = compute_z(inp);
    return {z, {}};
  }

  DIM_TEMPLATED INPUT compute_delta(
      const OUTPUT &delta, [[maybe_unused]] const CACHE &z_cache) const {
    INPUT new_delta;
    for (size_t ch = 0; ch < Ch_in; ++ch) {
      for (size_t r = 0; r < R_in; ++r) {
        for (size_t c = 0; c < C_in; ++c) {
          new_delta[ch][r, c] = delta[ch * R_in * C_in + r * C_in + c];
        }
      }
    }
    return new_delta;
  }
};

#undef DIM_TEMPLATED
#undef INPUT
#undef OUTPUT
#undef CACHE

#endif // ML_FLATTEN_H
