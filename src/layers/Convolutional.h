#ifndef ML_CONVOLUTIONAL_H
#define ML_CONVOLUTIONAL_H
#include "../lr_optimizer/Adam.h"
#include "../matrix/3DTensor.h"
#include "../matrix/AlgebraicTuple.h"
#include "LayerConcepts.h"
#include "ParamsLayer.h"

#include <cstddef>

#define RC_TEMPLATED template <size_t R_in, size_t C_in>
#define KERNEL Tensor3D<Ch_in, R_k, C_k>
#define INPUT Tensor3D<Ch_in, R_in, C_in>
#define OUTPUT Tensor3D<num_ker, R_in - R_k + 1, C_in - C_k + 1>
#define OUTPUT_MATRIX Matrix<R_in - R_k + 1, C_in - C_k + 1>
#define CACHE Cache<R_in, C_in>

template <size_t Ch_in, size_t R_k, size_t C_k, size_t num_ker>
class Convolutional
    : public ParamsLayer<
          Convolutional<Ch_in, R_k, C_k, num_ker>,
          AlgebraicTuple<StaticArray<num_ker, KERNEL>, Vector<num_ker>>> {

private:
  RC_TEMPLATED
  struct Cache {
    INPUT a_previous;
    OUTPUT z_cache;
  };

  using Kernel = Tensor3D<Ch_in, R_k, C_k>;
  using Bias = Vector<num_ker>;
  using Gradient =
      AlgebraicTuple<StaticArray<num_ker, KERNEL>, Vector<num_ker>>;

public:
  Convolutional(ActivationFunctionEnum activation_function_type,
                const AdamFactory &optimizer_factory)
      : ParamsLayer<Convolutional, Gradient>(activation_function_type,
                                             optimizer_factory) {}

  Convolutional(Gradient params,
                ActivationFunctionEnum activation_function_type,
                const AdamFactory &optimizer_factory)
      : ParamsLayer<Convolutional, Gradient>(params, activation_function_type,
                                             optimizer_factory) {}

  StaticArray<num_ker, Kernel> get_kernels() const {
    return this->params.template get<0>();
  }
  Bias get_bias() const { return this->params.template get<1>(); }

  StaticArray<num_ker, Kernel> get_kernel_gradients() const {
    return this->gradient.template get<0>();
  }
  Bias get_bias_gradients() const { return this->gradient.template get<1>(); }

  RC_TEMPLATED
  OUTPUT compute_z(const Tensor3D<Ch_in, R_in, C_in> &inp) const
    requires(R_in >= R_k && C_in >= C_k)
  {
    StaticArray<num_ker, Kernel> kernels = get_kernels();
    Bias bias = get_bias();

    OUTPUT result;
    for (size_t i = 0; i < num_ker; ++i) {
      result[i] = Matrix<R_in - R_k + 1, C_in - C_k + 1>(bias[i]);
      for (size_t ch = 0; ch < Ch_in; ++ch) {
        result[i] += inp[ch].valid_cross_correlation(kernels[i][ch]);
      }
    }
    return result;
  }

  RC_TEMPLATED
  ForwardCachedResult<OUTPUT, CACHE>
  compute_z_cached(const Tensor3D<Ch_in, R_in, C_in> &inp) const
    requires(R_in >= R_k && C_in >= C_k)
  {
    OUTPUT z = compute_z(inp);
    return {z, CACHE{inp, z}};
  }

  RC_TEMPLATED
  BackwardResult<INPUT, Gradient> compute_delta(const OUTPUT &delta,
                                                const CACHE &cache) const {
    const OUTPUT dz =
        delta.hadamard(this->activation_function->derivative(cache.z_cache));
    StaticArray<num_ker, Kernel> kernels = get_kernels();

    auto kernelGradients = StaticArray<num_ker, Kernel>();
    Bias biasGradients = Bias::zeros();

    for (size_t i = 0; i < num_ker; i++) {
      kernelGradients[i] =
          matrix_tensor_convolution<R_in, C_in>(cache.a_previous, dz[i]);
      biasGradients[i] = dz[i].sum();
    }

    INPUT next_delta = INPUT::zeros();
    for (size_t i = 0; i < num_ker; ++i) {
      for (size_t ch = 0; ch < Ch_in; ++ch) {
        next_delta[ch] += dz[i].full_convolution(kernels[i][ch]);
      }
    }

    return {next_delta, Gradient{kernelGradients, biasGradients}};
  }

private:
  template <size_t R_in, size_t C_in>
  KERNEL matrix_tensor_convolution(const INPUT &input,
                                   const OUTPUT_MATRIX &delta) const {
    KERNEL result;
    for (size_t ch = 0; ch < Ch_in; ++ch) {
      result[ch] = input[ch].valid_cross_correlation(delta);
    }
    return result;
  }
};

#undef RC_TEMPLATED
#undef KERNEL
#undef INPUT
#undef OUTPUT
#undef OUTPUT_MATRIX
#undef CACHE

#endif // ML_CONVOLUTIONAL_H
