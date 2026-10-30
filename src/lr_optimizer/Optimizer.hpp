#ifndef ML_OPTIMIZER_H
#define ML_OPTIMIZER_H

#include "../matrix/Algebraic.hpp"
#include <concepts>
#include <memory>
#include <nlohmann/json.hpp>

template <Algebraic Shape> class Optimizer {
  using json = nlohmann::json;

public:
  Optimizer(Shape &params, const double lr)
      : params(params), learning_rate(lr) {}
  virtual ~Optimizer() = default;

  virtual void adjust(const Shape &gradient) = 0;
  virtual json convert_to_json() const = 0;
  virtual void convert_from_json(const json &j) = 0;

  double get_learning_rate() const { return learning_rate; }

  friend void to_json(json &j, const Optimizer &opt) {
    j = opt.convert_to_json();
    j["learning_rate"] = opt.learning_rate;
  }

  friend void from_json(const json &j, Optimizer &opt) {
    opt.convert_from_json(j);
    j.at("learning_rate").get_to(opt.learning_rate);
  }

protected:
  Shape &params;
  double learning_rate;
};

template <typename F, typename Shape>
concept OptimizerFactoryFor =
    Algebraic<Shape> && requires(const F &f, Shape &params) {
      {
        f.build(params)
      } -> std::convertible_to<std::unique_ptr<Optimizer<Shape>>>;
    };

// template <typename F>
// concept OptimizerFactory =
//     requires(const F &f, Algebraic &params) {
//       {
//         f.build(params)
//       } -> std::convertible_to<std::unique_ptr<Optimizer<Shape>>>;
//     };

#endif // ML_OPTIMIZER_H
