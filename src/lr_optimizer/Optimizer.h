#ifndef ML_OPTIMIZER_H
#define ML_OPTIMIZER_H

#include "../matrix/Algebraic.h"

template <Algebraic Shape> class Optimizer {
public:
  Optimizer(Shape &params, const double lr)
      : params(params), learning_rate(lr) {}
  virtual ~Optimizer() = default;
  virtual void adjust(const Shape &gradient) = 0;

  double get_learning_rate() const { return learning_rate; }

protected:
  Shape &params;
  double learning_rate;
};

#endif // ML_OPTIMIZER_H
