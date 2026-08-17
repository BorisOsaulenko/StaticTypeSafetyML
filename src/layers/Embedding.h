//
// Created by Boris Osaulenko on 09/08/2026.
//

#ifndef ML_EMBEDDING_H
#define ML_EMBEDDING_H
#include "../matrix/Matrix.h"
#include "ParamsLayer.h"

#include <cstddef>

template <std::size_t TokenAmt, std::size_t EmbeddingDim,
          std::size_t ContextLength>
class Embedding
    : public ParamsLayer<Embedding<TokenAmt, EmbeddingDim, ContextLength>,
                         Matrix<TokenAmt, EmbeddingDim>> {
  using InputArray = StaticSpan<ContextLength, std::size_t>;
  using Input = StaticSpan<ContextLength, std::size_t>;
  using Output = StaticArray<ContextLength, Vector<EmbeddingDim>>;
  using Grad = Matrix<TokenAmt, EmbeddingDim>;
  using BackwardResult = LayerBackwardResult<Input, Grad>;

public:
  Embedding(ActivationFunctionEnum activation_function_type,
            const AdamFactory &optimizer_factory)
      : ParamsLayer<Embedding, Grad>(activation_function_type,
                                     optimizer_factory) {}
  Embedding(Grad &params, ActivationFunctionEnum activation_function_type,
            const AdamFactory &optimizer_factory)
      : ParamsLayer<Embedding, Grad>(params, activation_function_type,
                                     optimizer_factory) {}

  Output compute_z(const Input &inp) const {
    StaticArray<ContextLength, Vector<EmbeddingDim>> result;
    for (std::size_t tokenIdx = 0; tokenIdx < ContextLength; ++tokenIdx) {
      Vector<EmbeddingDim> vec;
      for (std::size_t colIdx = 0; colIdx < EmbeddingDim; ++colIdx) {
        vec[colIdx] = this->params[inp[tokenIdx]][colIdx];
      }
      result[tokenIdx] = vec;
    }
    return result;
  }
  BackwardResult
  compute_delta(const Output &delta, const Input &a_previous,
                [[maybe_unused]] const Output &z_cache) const {
    Grad grad = Grad::zeros();
    for (std::size_t tokenIdx = 0; tokenIdx < ContextLength; ++tokenIdx) {
      for (std::size_t colIdx = 0; colIdx < EmbeddingDim; ++colIdx) {
        grad[a_previous[tokenIdx]][colIdx] += delta[tokenIdx][colIdx];
      }
    }
    return BackwardResult{InputArray::zeros(), grad};
  }
};

#endif // ML_EMBEDDING_H
