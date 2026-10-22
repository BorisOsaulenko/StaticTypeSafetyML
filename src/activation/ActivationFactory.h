#ifndef ML_ACTIVATIONFACTORY_H
#define ML_ACTIVATIONFACTORY_H
#include "Identity.h"
#include "ReLu.h"
#include "Sigmoid.h"

#include <memory>
#include <stdexcept>

enum class ActivationFunctionEnum;
class ActivationFactory {
public:
  static std::unique_ptr<ActivationFunction>
  create(const ActivationFunctionEnum &type) {
    switch (type) {
    case ActivationFunctionEnum::Identity:
      return std::make_unique<Identity>();
    case ActivationFunctionEnum::ReLu:
      return std::make_unique<ReLu>();
    case ActivationFunctionEnum::Sigmoid:
      return std::make_unique<Sigmoid>();
    default:
      throw std::invalid_argument("Unknown activation function type");
    }
  }
};

#endif // ML_ACTIVATIONFACTORY_H
