//
// Created by Boris Osaulenko on 08/07/2026.
//

#ifndef ML_LOSSFACTORY_H
#define ML_LOSSFACTORY_H
#include "../matrix/Algebraic.h"
#include "CrossEntropy.h"
#include "Loss.h"
#include "MSE.h"

#include <stdexcept>

class LossFactory {
public:
  template <Algebraic T>
  static std::unique_ptr<Loss<T>> create(const LossEnum loss_type) {
    switch (loss_type) {
    case LossEnum::MSE:
      return std::make_unique<MSE<T>>();
    case LossEnum::CrossEntropy:
      return std::make_unique<CrossEntropy<T>>();
    default:
      throw std::invalid_argument("Unknown loss type");
    }
  }
};

#endif // ML_LOSSFACTORY_H
