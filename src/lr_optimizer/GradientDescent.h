#ifndef ML_GRADIENT_DESCENT
#define ML_GRADIENT_DESCENT

#include "Optimizer.h"
#include <memory>
#include <nlohmann/json.hpp>

template <Algebraic Shape> class SGD : public Optimizer<Shape> {
  using json = nlohmann::json;

public:
  SGD(Shape &params, double learning_rate)
      : Optimizer<Shape>(params, learning_rate) {}

  void adjust(const Shape &gradient) override {
    this->params -= gradient * this->learning_rate;
  }

  json convert_to_json() const { return {}; }
  void convert_from_json(const json &) {}
};

class SGDFactory {
public:
  explicit SGDFactory(const double learning_rate = 1e-3)
      : learning_rate(learning_rate) {}

  template <Algebraic Shape>
  std::unique_ptr<SGD<Shape>> build(Shape &initial_params) const {
    return std::make_unique<SGD<Shape>>(SGD(initial_params, learning_rate));
  }

private:
  double learning_rate;
};

#endif // ML_GRADIENT_DESCENT
