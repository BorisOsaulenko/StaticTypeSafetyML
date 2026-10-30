#include "../../src/matrix/Matrix.hpp"
#include "LayerConcepts.hpp"
#include "ParamsLayer.hpp"
#include <cmath>
#include <cstddef>

#define PARAMS Matrix<2, Emb> // row 0 for gamma, row 1 for beta
#define INPUT_LENGTH_TEMPLATED template <size_t R>

template <size_t Emb>
class AddAndNorm : public ParamsLayer<AddAndNorm<Emb>, PARAMS> {
  INPUT_LENGTH_TEMPLATED struct Cache {
    Matrix<R, Emb> a_prev;
    Vector<R> means;
    Vector<R> variances_inv;
  };

public:
  AddAndNorm()
      : ParamsLayer<AddAndNorm<Emb>, Vector<2>>(
            ActivationFunctionEnum::Identity) {}
  AddAndNorm(PARAMS &params)
      : ParamsLayer<AddAndNorm<Emb>, Vector<2>>(
            params, ActivationFunctionEnum::Identity) {}

  const double eps = 1e-5;
  const Vector<Emb> &get_gamma() { return this->params.get_row(0); }
  const Vector<Emb> &get_beta() { return this->params.get_row(1); }

  INPUT_LENGTH_TEMPLATED
  Matrix<R, Emb> compute_z(const Matrix<R, Emb> &inp) const {
    return forward_impl<R, Emb, false>(inp);
  }

  INPUT_LENGTH_TEMPLATED
  ForwardCachedResult<Matrix<R, Emb>, Cache<R>>
  compute_z_cached(const Matrix<R, Emb> &inp) const {
    return forward_impl<R, Emb, true>(inp);
  }

  INPUT_LENGTH_TEMPLATED
  BackwardResult<Matrix<R, Emb>, Vector<2>>
  compute_delta(const Matrix<R, Emb> &delta, const Cache<R> &cache) {
    const Vector<Emb> gamma = get_gamma();
    Vector<Emb> d_gamma = Vector<Emb>::zeros();
    Vector<Emb> d_beta = Vector<Emb>::zeros();

    const Matrix<R, Emb> &a_prev = cache.a_prev;
    const Vector<R> &means = cache.means;
    const Vector<R> &variances_inv = cache.variances_inv;

    Matrix<R, Emb> new_delta;

    // d_beta
    for (size_t row = 0; row < R; ++row) {
      for (size_t col = 0; col < Emb; ++col) {
        d_beta[col] += delta[row, col];
      }
    }

    // d_gamma
    for (size_t row = 0; row < R; ++row) {
      for (size_t col = 0; col < Emb; ++col) {
        d_gamma[col] += (a_prev[row, col] - means[row]) * variances_inv[row] *
                        delta[row, col];
      }
    }

    for (size_t row = 0; row < R; ++row) {
      for (size_t col = 0; col < Emb; ++col) {
      }
    }

    Vector<2> grad;
    grad[0] = d_gamma;
    grad[1] = d_beta;
    return {new_delta, grad};
  }

private:
  INPUT_LENGTH_TEMPLATED Vector<R> get_means(const Matrix<R, Emb> &inp) {
    Vector<R> result;
    for (size_t r = 0; r < R; ++r) {
      double mean = 0;
      for (size_t c = 0; c < Emb; ++c) {
        mean += inp[r, c];
      }
      result[r] = mean / Emb;
    }

    return result;
  }

  INPUT_LENGTH_TEMPLATED
  Vector<R> get_variances_inv(const Matrix<R, Emb> &inp,
                              const Vector<R> &mean) {
    Vector<R> result;
    for (size_t r = 0; r < R; ++r) {
      double sum = 0;
      for (size_t c = 0; c < Emb; ++c) {
        double tmp = inp[r, c] - mean[r];
        sum += tmp * tmp;
      }
      result[r] = 1 / std::sqrt(eps + sum / Emb);
    }
    return result;
  }

  // row-wise
  template <size_t R, size_t C, bool IsCached>
  auto forward_impl(const Matrix<R, C> &inp) {
    Vector<R> means = get_means(inp);
    Vector<R> variances_inv = get_variances_inv(inp, means);

    Matrix<R, C> result;
    for (size_t r = 0; r < R; ++r) {
      for (size_t c = 0; c < C; ++c) {
        double tmp = (inp[r, c] - means[r]) * variances_inv[r];
        result[r, c] = get_gamma()[c] * tmp + get_beta()[c];
      }
    }

    if constexpr (IsCached) {
      return ForwardCachedResult<Matrix<R, C>, Cache<R>>{
          result, {inp, means, variances_inv}};
    }

    return result;
  }
};