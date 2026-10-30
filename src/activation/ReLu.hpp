#ifndef ML_RELU_H
#define ML_RELU_H
#include "ActivationFunction.hpp"

class ReLu : public ActivationFunction {
public:
  ReLu() : ActivationFunction(ActivationFunctionEnum::ReLu) {}

  [[nodiscard]] value_type derivative(const value_type x) const override {
    return x > 0 ? 1 : 0;
  }

  [[nodiscard]] value_type activate(const value_type x) const override {
    return x > 0 ? x : 0;
  }
};

#endif // ML_RELU_H
