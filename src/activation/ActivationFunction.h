//
// Created by Boris Osaulenko on 05/07/2026.
//
#pragma once

#ifndef ML_ACTIVATIONFUNCTION_H
#define ML_ACTIVATIONFUNCTION_H
#include "../matrix/Algebraic.h"

enum class ActivationFunctionEnum {
  Identity,
  ReLu,
  Sigmoid,
};

class ActivationFunction {
protected:
  using value_type = double;
  ActivationFunctionEnum type;

public:
  explicit ActivationFunction(const ActivationFunctionEnum type) : type(type) {}

  [[nodiscard]] ActivationFunctionEnum get_type() const { return type; }

  [[nodiscard]] virtual value_type activate(value_type x) const = 0;

  [[nodiscard]] virtual value_type derivative(value_type x) const = 0;

  template <Algebraic Shape> [[nodiscard]] Shape activate(const Shape &x) {
    return x.elementwise(
        [this](const value_type val) { return activate(val); });
  }

  template <Algebraic Shape> [[nodiscard]] Shape derivative(const Shape &x) {
    return x.elementwise(
        [this](const value_type val) { return derivative(val); });
  }

  virtual ~ActivationFunction() = default;
};

#endif // ML_ACTIVATIONFUNCTION_H
