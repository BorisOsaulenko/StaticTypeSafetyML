//
// Created by Boris Osaulenko on 12/07/2026.
//

#ifndef ML_CONVOLUTIONAL_H
#define ML_CONVOLUTIONAL_H
#include "../lr_optimizer/Adam.h"
#include "../matrix/3DTensor.h"
#include "../matrix/AlgebraicTuple.h"
#include "ParamsLayer.h"

#include <nlohmann/json.hpp>

template <int Ch_in, int R_in, int C_in, int R_k, int C_k, int num_ker>
using ConvGradType =
    AlgebraicTuple<StaticArray<num_ker, Tensor3D<Ch_in, R_k, C_k>>,
                   Vector<num_ker>>;

template <int Ch_in, int R_in, int C_in, int R_k, int C_k, int num_ker>
  requires(R_in >= R_k && C_in >= C_k)
class Convolutional
    : public ParamsLayer<Convolutional<Ch_in, R_in, C_in, R_k, C_k, num_ker>,
                         ConvGradType<Ch_in, R_in, C_in, R_k, C_k, num_ker>> {

public:
  using Input = Tensor3D<Ch_in, R_in, C_in>;
  using Output = Tensor3D<num_ker, R_in - R_k + 1, C_in - C_k + 1>;

private:
  using Output_Matrix = Matrix<R_in - R_k + 1, C_in - C_k + 1>;
  using Kernel = Tensor3D<Ch_in, R_k, C_k>;
  using Bias = Vector<num_ker>;
  using value_type = double;
  using json = nlohmann::json;
  using Gradient = ConvGradType<Ch_in, R_in, C_in, R_k, C_k, num_ker>;
  using BackwardResult = LayerBackwardResult<Input, Gradient>;

public:
  Convolutional(ActivationFunctionEnum activation_function_type,
                AdamFactory &optimizer_factory)
      : ParamsLayer<Convolutional, Gradient>(activation_function_type,
                                             optimizer_factory) {}

  Convolutional(Gradient params,
                ActivationFunctionEnum activation_function_type,
                AdamFactory &optimizer_factory)
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

  Output compute_z(const Input &inp) const {
    StaticArray<num_ker, Kernel> kernels = get_kernels();
    Bias bias = get_bias();

    Output result;
    for (std::size_t i = 0; i < num_ker; ++i) {
      result[i] = Matrix<R_in - R_k + 1, C_in - C_k + 1>(bias[i]);
      for (std::size_t ch = 0; ch < Ch_in; ++ch) {
        result[i] += inp[ch].valid_cross_correlation(kernels[i][ch]);
      }
    }
    return result;
  }

  BackwardResult compute_delta(const Output &delta, const Input &a_previous,
                               const Output &z_cache) const {
    const Output dz =
        delta.hadamard(this->activation_function->derivative(z_cache));
    StaticArray<num_ker, Kernel> kernels = get_kernels();

    auto kernelGradients = StaticArray<num_ker, Kernel>();
    Bias biasGradients = Bias::zeros();

    for (std::size_t i = 0; i < num_ker; i++) {
      kernelGradients[i] = matrix_tensor_convolution(a_previous, dz[i]);
      biasGradients[i] = dz[i].sum();
    }

    Input next_delta = Input::zeros();
    for (std::size_t i = 0; i < num_ker; ++i) {
      for (std::size_t ch = 0; ch < Ch_in; ++ch) {
        next_delta[ch] += dz[i].full_convolution(kernels[i][ch]);
      }
    }

    return BackwardResult{next_delta, Gradient{kernelGradients, biasGradients}};
  }

private:
  Kernel matrix_tensor_convolution(const Input &input,
                                   const Output_Matrix &delta) const {
    Kernel result;
    for (std::size_t ch = 0; ch < Ch_in; ++ch) {
      result[ch] = input[ch].valid_cross_correlation(delta);
    }
    return result;
  }
};

#endif // ML_CONVOLUTIONAL_H
