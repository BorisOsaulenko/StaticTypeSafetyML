#ifndef ML_MSE_H
#define ML_MSE_H
#include "../matrix/Algebraic.hpp"
#include "Loss.hpp"

template <Algebraic T> class MSE : public Loss<T> {
  using Loss<T>::Loss;

public:
  explicit MSE() : Loss<T>(LossEnum::MSE) {}
  double compute(T y_pred, T y_true) override {
    return (y_pred - y_true).dot(y_pred - y_true) / y_true.size();
  }

  T derivative(T y_pred, T y_true) override {
    return (y_pred - y_true) / y_true.size() * 2;
  }
};

#endif // ML_MSE_H
