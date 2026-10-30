#ifndef ML_EMBEDDING_H
#define ML_EMBEDDING_H
#include "../matrix/Matrix.hpp"
#include "LayerConcepts.hpp"
#include "ParamsLayer.hpp"

#include <cstddef>

#define LENGTH_TEMPLATED template <size_t InputLength>
#define INPUT std::span<size_t, InputLength>
#define OUTPUT Matrix<InputLength, EmbeddingDim>
#define CACHE Cache<InputLength>

template <size_t TokenAmt, size_t EmbeddingDim>
class Embedding : public ParamsLayer<Embedding<TokenAmt, EmbeddingDim>,
                                     Matrix<TokenAmt, EmbeddingDim>> {

  LENGTH_TEMPLATED using Input = std::span<size_t, InputLength>;
  LENGTH_TEMPLATED using Output = Matrix<InputLength, EmbeddingDim>;
  using Grad = Matrix<TokenAmt, EmbeddingDim>;

  LENGTH_TEMPLATED struct Cache {
    INPUT a_prev;
  };

public:
  Embedding(ActivationFunctionEnum activation_function_type,
            const OptimizerFactoryFor<Grad> auto &optimizer_factory)
      : ParamsLayer<Embedding, Grad>(activation_function_type,
                                     optimizer_factory) {}
  Embedding(Grad &params, ActivationFunctionEnum activation_function_type,
            const OptimizerFactoryFor<Grad> auto &optimizer_factory)
      : ParamsLayer<Embedding, Grad>(params, activation_function_type,
                                     optimizer_factory) {}

  LENGTH_TEMPLATED
  OUTPUT compute_z(const INPUT &inp) const {
    OUTPUT result;
    for (size_t embIdx = 0; embIdx < EmbeddingDim; ++embIdx) {
      for (size_t tokenIdx = 0; tokenIdx < InputLength; ++tokenIdx) {
        assert(inp[tokenIdx] < TokenAmt);
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
        grad[cache.a_prev[tokenIdx], colIdx] += delta[tokenIdx, colIdx];
      }
    }
    return {INPUT{}, grad};
  }
};

#undef LENGTH_TEMPLATED
#undef INPUT
#undef OUTPUT
#undef CACHE

#endif // ML_EMBEDDING_H
