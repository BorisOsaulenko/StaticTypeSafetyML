#ifndef ML_ADAM_H
#define ML_ADAM_H
#include "Optimizer.h"
#include <memory>
#include <nlohmann/json.hpp>

template <Algebraic Shape> class Adam : public Optimizer<Shape> {
  using json = nlohmann::json;

public:
  Adam(Shape &params, double learning_rate, double beta1, double beta2,
       double epsilon)
      : Optimizer<Shape>(params, learning_rate) {
    this->beta1 = this->currBeta1 = beta1;
    this->beta2 = this->currBeta2 = beta2;
    this->epsilon = epsilon;
  }

  void adjust(const Shape &gradient) override {
    t++;
    m = m * beta1 + gradient * (1 - beta1);
    v = v * beta2 + gradient.hadamard(gradient) * (1 - beta2);

    Shape m_hat = m * (1.0 / (1.0 - currBeta1));
    Shape v_hat = v * (1.0 / (1.0 - currBeta2));

    currBeta1 *= beta1;
    currBeta2 *= beta2;

    this->params -= m_hat / (v_hat.sqrt() + epsilon) * this->learning_rate;
  }

  json convert_to_json() const {
    return {{"beta1", beta1},
            {"currBeta1", currBeta1},
            {"beta2", beta2},
            {"currBeta2", currBeta2},
            {"epsilon", epsilon},
            {"t", t},
            {"m", m},
            {"v", v}};
  }

  void convert_from_json(const nlohmann::json &j) {
    j.at("beta1").get_to(beta1);
    j.at("currBeta1").get_to(currBeta1);
    j.at("beta2").get_to(beta2);
    j.at("currBeta2").get_to(currBeta2);
    j.at("epsilon").get_to(epsilon);
    j.at("t").get_to(t);
    j.at("m").get_to(m);
    j.at("v").get_to(v);
  }

private:
  double beta1, currBeta1;
  double beta2, currBeta2;
  double epsilon;

  long long t = 0;

  Shape m;
  Shape v;
};

class AdamFactory {
public:
  explicit AdamFactory(const double learning_rate = 1e-3,
                       const double beta1 = 0.9, const double beta2 = 0.999,
                       const double epsilon = 1e-8)
      : learning_rate(learning_rate), beta1(beta1), beta2(beta2),
        epsilon(epsilon) {}

  template <Algebraic T>
  std::unique_ptr<Optimizer<T>> build(T &initial_params) const {
    return std::make_unique<Adam<T>>(initial_params, learning_rate, beta1,
                                     beta2, epsilon);
  }

private:
  double learning_rate;
  double beta1;
  double beta2;
  double epsilon;
};

#endif // ML_ADAM_H
