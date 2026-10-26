#ifndef ML_EMBEDDING_H
#define ML_EMBEDDING_H
#include "../matrix/Matrix.h"
#include "LayerConcepts.h"
#include "ParamsLayer.h"

#include <cstddef>

#define LENGTH_TEMPLATED template <size_t InputLength>
#define INPUT StaticSpan<InputLength, size_t>
#define OUTPUT Matrix<InputLength, EmbeddingDim>
#define CACHE Cache<InputLength>

template <size_t TokenAmt, size_t EmbeddingDim>
class Embedding : public ParamsLayer<Embedding<TokenAmt, EmbeddingDim>,
                                     Matrix<TokenAmt, EmbeddingDim>> {

  LENGTH_TEMPLATED using Input = StaticSpan<InputLength, size_t>;
  LENGTH_TEMPLATED using Output = Matrix<InputLength, EmbeddingDim>;
  using Grad = Matrix<TokenAmt, EmbeddingDim>;

  LENGTH_TEMPLATED struct Cache {
    INPUT a_prev;
  };

public:
  Embedding()
      : ParamsLayer<Embedding, Grad>(ActivationFunctionEnum::Identity) {}
  Embedding(Grad &params)
      : ParamsLayer<Embedding, Grad>(params, ActivationFunctionEnum::Identity) {
  }

  LENGTH_TEMPLATED
  OUTPUT compute_z(const INPUT &inp) const {
    OUTPUT result;
    for (size_t embIdx = 0; embIdx < EmbeddingDim; ++embIdx) {
      for (size_t tokenIdx = 0; tokenIdx < InputLength; ++tokenIdx) {
        result[tokenIdx, embIdx] = this->params[inp[tokenIdx], embIdx];
      }
    }
    return result;
  }

  LENGTH_TEMPLATED
  ForwardCachedResult<OUTPUT, CACHE> compute_z_cached(const INPUT &inp) const {
    const OUTPUT z = compute_z(inp);
    return {z, inp};
  }

  LENGTH_TEMPLATED
  BackwardResult<INPUT, Grad> compute_delta(const OUTPUT &delta,
                                            const CACHE cache) const {
    Grad grad = Grad::zeros();
    for (size_t tokenIdx = 0; tokenIdx < InputLength; ++tokenIdx) {
      for (size_t colIdx = 0; colIdx < EmbeddingDim; ++colIdx) {
        grad[cache.a_prev[tokenIdx]][colIdx] += delta[tokenIdx][colIdx];
      }
    }
    return {INPUT::zeros(), grad};
  }
};

#undef LENGTH_TEMPLATED
#undef INPUT
#undef OUTPUT
#undef CACHE

#endif // ML_EMBEDDING_H
