//
// Created by Boris Osaulenko on 06/07/2026.
//

#ifndef ML_SIGMOID_H
#define ML_SIGMOID_H
#include "ActivationFunction.h"
#include <cmath>

class Sigmoid : public ActivationFunction {
public:
  Sigmoid() : ActivationFunction(ActivationFunctionEnum::Sigmoid) {}

  [[nodiscard]] value_type derivative(const value_type x) const override {
    const value_type sigmoid_x = activate(x);
    return sigmoid_x * (1 - sigmoid_x);
  }

  [[nodiscard]] value_type activate(const value_type x) const override {
    return 1 / (1 + std::exp(-x));
  }
};

#endif // ML_SIGMOID_H
