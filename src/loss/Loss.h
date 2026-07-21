//
// Created by Boris Osaulenko on 08/07/2026.
//

#ifndef ML_LOSS_H
#define ML_LOSS_H
#include "../matrix/AlgebraicSpan.h"

enum class LossEnum { MSE, CrossEntropy };

template <Algebraic T> class Loss {
private:
  LossEnum type;

public:
  virtual ~Loss() = default;
  explicit Loss(const LossEnum t) : type(t) {}

  virtual double compute(T y_pred, T y_true) = 0;
  virtual T derivative(T y_pred, T y_true) = 0;

  template <std::size_t SIZE>
  AlgebraicArray<SIZE, double>
  compute(const AlgebraicSpan<SIZE, const T> &y_pred,
          const AlgebraicSpan<SIZE, const T> &y_true) {
    AlgebraicArray<SIZE, double> losses;
    for (std::size_t i = 0; i < SIZE; ++i)
      losses[i] = compute(y_pred[i], y_true[i]);
    return losses;
  }

  template <std::size_t SIZE>
  AlgebraicArray<SIZE, T>
  derivative(const AlgebraicArray<SIZE, T> &y_pred,
             const AlgebraicSpan<SIZE, const T> &y_true) {
    AlgebraicArray<SIZE, T> deltas;
    for (std::size_t i = 0; i < SIZE; ++i)
      deltas[i] = derivative(y_pred[i], y_true[i]);
    return deltas;
  }
};

#endif // ML_LOSS_H