#ifndef ML_IDENTITY_H
#define ML_IDENTITY_H
#include "ActivationFunction.h"

class Identity : public ActivationFunction {
public:
  explicit Identity() : ActivationFunction(ActivationFunctionEnum::Identity) {}

  [[nodiscard]] value_type
  derivative(const value_type __unused) const override {
    return 1;
  }

  [[nodiscard]] value_type activate(const value_type x) const override {
    return x;
  }
};

#endif // ML_IDENTITY_H
