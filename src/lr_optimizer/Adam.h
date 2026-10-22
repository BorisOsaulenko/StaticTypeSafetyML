#ifndef ML_ADAM_H
#define ML_ADAM_H
#include "Optimizer.h"

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

  friend void to_json(nlohmann::json &j, const Adam &adam) {
    j = json{{"beta1", adam.beta1},
             {"currBeta1", adam.currBeta1},
             {"beta2", adam.beta2},
             {"currBeta2", adam.currBeta2},
             {"epsilon", adam.epsilon},
             {"t", adam.t},
             {"m", adam.m},
             {"v", adam.v}};
  }

  friend void from_json(const nlohmann::json &j, Adam &adam) {
    j.at("beta1").get_to(adam.beta1);
    j.at("currBeta1").get_to(adam.currBeta1);
    j.at("beta2").get_to(adam.beta2);
    j.at("currBeta2").get_to(adam.currBeta2);
    j.at("epsilon").get_to(adam.epsilon);
    j.at("t").get_to(adam.t);
    j.at("m").get_to(adam.m);
    j.at("v").get_to(adam.v);
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

  template <Algebraic T> Adam<T> build(T &initial_params) const {
    return Adam<T>(initial_params, learning_rate, beta1, beta2, epsilon);
  }

private:
  double learning_rate;
  double beta1;
  double beta2;
  double epsilon;
};

#endif // ML_ADAM_H
