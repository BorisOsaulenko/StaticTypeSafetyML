//
// Created by Boris Osaulenko on 11/08/2026.
//

#ifndef ML_ATTENTIONHEAD_H
#define ML_ATTENTIONHEAD_H
#include "../matrix/AlgebraicTuple.h"
#include "../matrix/Matrix.h"
#include "ParamsLayer.h"

#include <cstddef>

template <std::size_t EmbeddingDim, std::size_t HeadDim>
class AttentionHead
    : public ParamsLayer<
          AttentionHead<EmbeddingDim, HeadDim>,
          StaticArray<4, Matrix<HeadDim, EmbeddingDim>>> { // query, key,
                                                           // value_down,
                                                           // value_up
  using Grad = StaticArray<4, Matrix<HeadDim, EmbeddingDim>>;
  using json = nlohmann::json;

  template <std::size_t ContextLength>
  using Input = Matrix<EmbeddingDim, ContextLength>;
  template <std::size_t ContextLength>
  using Output = Matrix<EmbeddingDim, ContextLength>;
  template <std::size_t ContextLength>
  using BackwardResult = LayerBackwardResult<Input<ContextLength>, Grad>;

  double neg_inf = std::numeric_limits<double>::lowest();

public:
  explicit AttentionHead(const ActivationFunctionEnum activation_function_type,
                         const AdamFactory &optimizer_factory)
      : ParamsLayer<AttentionHead, Grad>(activation_function_type,
                                         optimizer_factory) {}

  explicit AttentionHead(Grad &params,
                         const ActivationFunctionEnum activation_function_type,
                         const AdamFactory &optimizer_factory)
      : ParamsLayer<AttentionHead, Grad>(params, activation_function_type,
                                         optimizer_factory) {}

  Matrix<HeadDim, EmbeddingDim> get_query_weights() const {
    return this->params.template get<0>();
  }
  Matrix<HeadDim, EmbeddingDim> get_key_weights() const {
    return this->params.template get<1>();
  }
  Matrix<HeadDim, EmbeddingDim> get_value_down_weights() const {
    return this->params.template get<2>();
  }
  Matrix<HeadDim, EmbeddingDim> get_value_up_weights() const {
    return this->params.template get<3>();
  }

  template <std::size_t ContextLength>
  Output<ContextLength> compute_z(const Input<ContextLength> &inp) const {
    const Matrix<HeadDim, EmbeddingDim> query_weights = get_query_weights();
    const Matrix<HeadDim, EmbeddingDim> key_weights = get_key_weights();
    Matrix<HeadDim, ContextLength> queries = query_weights * inp;
    Matrix<HeadDim, ContextLength> keys = key_weights * inp;

    Matrix<ContextLength, ContextLength> attention_scores =
        keys.transpose() * queries / std::sqrt(HeadDim);

    // set later words influence to -inf
    for (std::size_t row = 0; row < ContextLength; ++row) {
      for (std::size_t col = 0; col < row; ++col) {
        attention_scores[row, col] = neg_inf;
      }
    }

    // col wise softmax
    for (std::size_t col = 0; col < ContextLength; ++col) {
      attention_scores.set_col(col, softmax(attention_scores.get_col(col)));
    }

    const auto &value_down_weights = get_value_down_weights();
    const auto &value_up_weights = get_value_up_weights();
    Matrix<EmbeddingDim, ContextLength> value_matrix =
        value_up_weights.transpose() * (value_down_weights * inp);

    return value_matrix * attention_scores;
  }

  template <std::size_t ContextLength>
  BackwardResult<ContextLength>
  compute_delta(const Output<ContextLength> &delta,
                const Input<ContextLength> &a_previous,
                [[maybe_unused]] const Output<ContextLength> &z_cache) const {

    Grad grad = Grad::zeros();
    Input<ContextLength> new_delta = Input<ContextLength>::zeros();
    return BackwardResult<ContextLength>{new_delta, grad};
  }

private:
  template <std::size_t ContextLength>
  static Vector<ContextLength> softmax(const Vector<ContextLength> &vec) {
    return vec.exp() / vec.exp().sum();
  }
};

#endif // ML_ATTENTIONHEAD_H
