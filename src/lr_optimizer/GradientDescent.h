#ifndef ML_GRADIENT_DESCENT
#define ML_GRADIENT_DESCENT
#include "../matrix/Matrix.h"
#include "Optimizer.h"

template <Algebraic Shape> class SGD : public Optimizer<Shape> {

public:
  SGD(Shape &params, double learning_rate)
      : Optimizer<Shape>(params, learning_rate) {}

  void adjust(const Shape &gradient) override {
    this->params -= gradient * this->learning_rate;
  }
};

class SGDFactory {
public:
  explicit SGDFactory(const double learning_rate = 1e-3)
      : learning_rate(learning_rate) {}

  template <Algebraic Shape> SGD<Shape> build(Shape &initial_params) const {
    return SGD<Shape>(initial_params, learning_rate);
  }

private:
  double learning_rate;
};

#endif // ML_GRADIENT_DESCENT
