#ifndef ML_ATTENTIONHEAD_H
#define ML_ATTENTIONHEAD_H

#include "../matrix/Matrix.h"
#include "ParamsLayer.h"

#include <cstddef>

#define LENGTH_TEMPLATED template <size_t InputLength>

template <size_t EmbeddingDim, size_t HeadDim>
class AttentionHead
    : public ParamsLayer<
          AttentionHead<EmbeddingDim, HeadDim>,
          StaticArray<4, Matrix<HeadDim, EmbeddingDim>>> { // query, key,
                                                           // value_down,
                                                           // value_up
  using Grad = StaticArray<4, Matrix<HeadDim, EmbeddingDim>>;

  LENGTH_TEMPLATED
  using Input = Matrix<EmbeddingDim, InputLength>;
  LENGTH_TEMPLATED
  using Output = Matrix<EmbeddingDim, InputLength>;
  LENGTH_TEMPLATED
  using BackwardResult = LayerBackwardResult<Input<InputLength>, Grad>;

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

  LENGTH_TEMPLATED
  Output<InputLength> compute_z(const Input<InputLength> &inp) const {
    const Matrix<HeadDim, EmbeddingDim> query_weights = get_query_weights();
    const Matrix<HeadDim, EmbeddingDim> key_weights = get_key_weights();
    Matrix<HeadDim, InputLength> queries = query_weights * inp;
    Matrix<HeadDim, InputLength> keys = key_weights * inp;

    Matrix<InputLength, InputLength> attention_scores =
        keys.transpose() * queries / std::sqrt(HeadDim);

    // set later words influence to -inf
    for (size_t row = 0; row < InputLength; ++row) {
      for (size_t col = 0; col < row; ++col) {
        attention_scores[row, col] = neg_inf;
      }
    }

    // col wise softmax
    for (size_t col = 0; col < InputLength; ++col) {
      attention_scores.set_col(col, softmax(attention_scores.get_col(col)));
    }

    const auto &value_down_weights = get_value_down_weights();
    const auto &value_up_weights = get_value_up_weights();
    Matrix<EmbeddingDim, InputLength> value_matrix =
        value_up_weights.transpose() * (value_down_weights * inp);

    return value_matrix * attention_scores;
  }

  LENGTH_TEMPLATED
  BackwardResult<InputLength>
  compute_delta(const Output<InputLength> &delta,
                const Input<InputLength> &a_previous,
                [[maybe_unused]] const Output<InputLength> &z_cache) const {

    Grad grad = Grad::zeros();
    Input<InputLength> new_delta = Input<InputLength>::zeros();
    return BackwardResult<InputLength>{new_delta, grad};
  }

private:
  LENGTH_TEMPLATED
  static Vector<InputLength> softmax(const Vector<InputLength> &vec) {
    return vec.exp() / vec.exp().sum();
  }
};

#undef LENGTH_TEMPLATED

#endif // ML_ATTENTIONHEAD_H
