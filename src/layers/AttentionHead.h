#ifndef ML_ATTENTIONHEAD_H
#define ML_ATTENTIONHEAD_H

#include "../matrix/Matrix.h"

#include "LayerConcepts.h"
#include "ParamsLayer.h"

#include <cmath>
#include <cstddef>

#define LENGTH_TEMPLATED template <size_t InputLength>
#define INPUT Matrix<InputLength, EmbeddingDim>
#define OUTPUT Matrix<InputLength, EmbeddingDim>
#define CACHE Cache<InputLength>
#define PARAM_MATRIX Matrix<EmbeddingDim, HeadDim>
#define GRAD                                                                   \
  StaticArray<4, PARAM_MATRIX> // query, key,
                               // value_down,
                               // value_up
#define BACKWARD_RESULT BackwardResult<INPUT, GRAD>

template <size_t EmbeddingDim, size_t HeadDim>
class AttentionHead
    : public ParamsLayer<AttentionHead<EmbeddingDim, HeadDim>, GRAD> {

  double neg_inf = std::numeric_limits<double>::lowest();

  LENGTH_TEMPLATED
  struct Cache {
    INPUT a_prev;
    Matrix<InputLength, HeadDim> queries, keys;
    Matrix<InputLength, InputLength> attention_softmaxed;
    Matrix<InputLength, EmbeddingDim> values_matrix;
  };

public:
  AttentionHead(const ActivationFunctionEnum activation_function_type)
      : ParamsLayer<AttentionHead, GRAD>(activation_function_type) {}

  AttentionHead(GRAD &params,
                const ActivationFunctionEnum activation_function_type)
      : ParamsLayer<AttentionHead, GRAD>(params, activation_function_type) {}

  const PARAM_MATRIX &get_query_weights() const { return this->params[0]; }
  const PARAM_MATRIX &get_key_weights() const { return this->params[1]; }
  const PARAM_MATRIX &get_value_down_weights() const { return this->params[2]; }
  const PARAM_MATRIX &get_value_up_weights() const { return this->params[3]; }

  LENGTH_TEMPLATED OUTPUT compute_z(const INPUT &inp) const {
    return forward_impl<false, InputLength>(inp);
  }
  LENGTH_TEMPLATED ForwardCachedResult<OUTPUT, CACHE>
  compute_z_cached(const INPUT &inp) const {
    return forward_impl<true, InputLength>(inp);
  }

  LENGTH_TEMPLATED
  BACKWARD_RESULT
  compute_delta(const OUTPUT &delta, const CACHE &cache) const {
    const Matrix<InputLength, InputLength> dL_dA_s =
        delta * cache.values_matrix.transpose();

    Vector<InputLength> X;
    for (size_t row = 0; row < InputLength; row++) {
      for (size_t col = 0; col < InputLength; col++) {
        X[row] += dL_dA_s[row, col] * cache.attention_softmaxed[row, col];
      }
    }

    Matrix<InputLength, InputLength> dL_dA;
    for (size_t row = 0; row < InputLength; row++) {
      for (size_t col = 0; col < InputLength; col++) {
        dL_dA[row, col] =
            cache.attention_softmaxed[row, col] * (dL_dA_s[row, col] - X[row]);
      }
    }

    const double head_dim_inverse = 1 / sqrt(HeadDim);
    const PARAM_MATRIX dWk = head_dim_inverse * cache.a_prev.transpose() *
                             dL_dA.transpose() * cache.queries;
    const PARAM_MATRIX dWq =
        head_dim_inverse * cache.a_prev.transpose() * dL_dA * cache.keys;
    const PARAM_MATRIX dWu = delta.transpose() * cache.attention_softmaxed *
                             cache.a_prev * this->get_value_down_weights();
    const PARAM_MATRIX dWd = cache.a_prev.transpose() *
                             cache.attention_softmaxed.transpose() * delta *
                             this->get_value_up_weights();

    const INPUT dInput_Q = head_dim_inverse * dL_dA * cache.keys *
                           this->get_query_weights().transpose();
    const INPUT dInput_K = head_dim_inverse * dL_dA.transpose() *
                           cache.queries * this->get_key_weights().transpose();
    const INPUT dInput_V = cache.attention_softmaxed.transpose() * delta *
                           this->get_value_up_weights() *
                           this->get_value_down_weights().transpose();
    const INPUT dInput = dInput_Q + dInput_K + dInput_V;
    return {dInput, {dWq, dWk, dWd, dWu}};
  }

private:
  LENGTH_TEMPLATED
  static Vector<InputLength> softmax(const Vector<InputLength> &vec) {
    return vec.exp() / vec.exp().sum();
  }

  template <bool WithCache, size_t InputLength>
  auto forward_impl(const INPUT &inp) const {
    Matrix<InputLength, HeadDim> queries = inp * get_query_weights();
    Matrix<InputLength, HeadDim> keys = inp * get_key_weights();

    Matrix<InputLength, InputLength> attention_scores =
        queries * keys.transpose() / std::sqrt(HeadDim);

    // set later words influence to -inf
    for (size_t row = 0; row < InputLength; ++row) {
      for (size_t col = row + 1; col < InputLength; ++col) {
        attention_scores[row, col] = neg_inf;
      }
    }

    // row wise softmax
    for (size_t row = 0; row < InputLength; ++row) {
      double sum = 0;
      for (size_t col = 0; col < InputLength; ++col)
        sum += std::exp(attention_scores[row, col]);
      for (size_t col = 0; col < InputLength; ++col)
        attention_scores[row, col] = std::exp(attention_scores[row, col]) / sum;
    }

    const auto &value_down_weights = get_value_down_weights();
    const auto &value_up_weights = get_value_up_weights();
    Matrix<InputLength, EmbeddingDim> value_matrix =
        inp * value_down_weights * value_up_weights.transpose();

    OUTPUT z = attention_scores * value_matrix;

    if constexpr (WithCache)
      return ForwardCachedResult<OUTPUT, CACHE>{
          std::move(z),
          CACHE{inp, std::move(queries), std::move(keys),
                std::move(attention_scores), std::move(value_matrix)}};
    else
      return z;
  }
};

#undef LENGTH_TEMPLATED
#undef INPUT
#undef OUTPUT
#undef CACHE
#undef PARAM_MATRIX
#undef GRAD
#undef BACKWARD_RESULT

#endif // ML_ATTENTIONHEAD_H
