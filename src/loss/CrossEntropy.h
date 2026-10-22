#ifndef ML_CROSSENTROPY_H
#define ML_CROSSENTROPY_H
#include "../matrix/Algebraic.h"
#include "Loss.h"

template <Algebraic T> class CrossEntropy : public Loss<T> {
  using Loss<T>::Loss;

private:
  T softmax(T y_pred) {
    T exp_values = y_pred.exp();
    return exp_values / exp_values.sum();
  }

public:
  explicit CrossEntropy() : Loss<T>(LossEnum::CrossEntropy) {}
  double compute(T y_pred, T y_expected) override {
    T y_softmax = softmax(y_pred);
    T log = y_softmax.log();
    return -y_expected.dot(log);
  }

  T derivative(T y_pred, T y_expected) override {
    T y_softmax = softmax(y_pred);
    return y_softmax - y_expected;
  }
};

#endif // ML_CROSSENTROPY_H
