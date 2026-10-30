#ifndef ML_EMBEDDING_H
#define ML_EMBEDDING_H
#include "../matrix/Matrix.hpp"
#include "LayerConcepts.hpp"
#include "ParamsLayer.hpp"

#include <cstddef>

#define LENGTH_TEMPLATED template <size_t InputLength>
#define INPUT std::span<size_t, InputLength>
#define OUTPUT Matrix<InputLength, EmbeddingDim>

template <size_t TokenAmt, size_t EmbeddingDim>
class Embedding : public ParamsLayer<Embedding<TokenAmt, EmbeddingDim>,
                                     Matrix<TokenAmt, EmbeddingDim>> {

  LENGTH_TEMPLATED using Input = std::span<size_t, InputLength>;
  LENGTH_TEMPLATED using Output = Matrix<InputLength, EmbeddingDim>;
  using Grad = Matrix<TokenAmt, EmbeddingDim>;

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
  Matrix<EmbeddingDim, InputLength> compute_z(const INPUT &inp) const {
    Matrix<EmbeddingDim, InputLength> result;
    for (size_t embIdx = 0; embIdx < EmbeddingDim; ++embIdx) {
      for (size_t tokenIdx = 0; tokenIdx < InputLength; ++tokenIdx) {
        result[embIdx][tokenIdx] = this->params[inp[tokenIdx]][embIdx];
      }
    }
    return result;
  }

  LENGTH_TEMPLATED
  BackwardResult<INPUT, Grad>
  compute_delta(const OUTPUT &delta, const INPUT &a_previous,
                [[maybe_unused]] const OUTPUT &z_cache) const {
    Grad grad = Grad::zeros();
    for (size_t tokenIdx = 0; tokenIdx < InputLength; ++tokenIdx) {
      for (size_t colIdx = 0; colIdx < EmbeddingDim; ++colIdx) {
        grad[a_previous[tokenIdx]][colIdx] += delta[tokenIdx][colIdx];
      }
    }
    return ForwardCachedResult{INPUT::zeros(), grad};
  }
};

#undef LENGTH_TEMPLATED
#undef INPUT
#undef OUTPUT

#endif // ML_EMBEDDING_H
