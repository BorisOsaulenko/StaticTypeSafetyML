#ifndef ML_ALGEBRAIC_H
#define ML_ALGEBRAIC_H

#include <concepts>
#include <nlohmann/json.hpp>

template <typename T>
concept Algebraic = requires(T a, T b, double scalar, nlohmann::json j) {
  { a + b } -> std::same_as<T>;
  { a += b } -> std::same_as<T &>;
  { a - b } -> std::same_as<T>;
  { a -= b } -> std::same_as<T &>;
  { a / b } -> std::same_as<T>; // element wise
  { a /= b } -> std::same_as<T &>;

  { a * scalar } -> std::same_as<T>;
  { a *= scalar } -> std::same_as<T &>;
  { a / scalar } -> std::same_as<T>;
  { a /= scalar } -> std::same_as<T &>;
  { a + scalar } -> std::same_as<T>;
  { a += scalar } -> std::same_as<T &>;
  { a - scalar } -> std::same_as<T>;
  { a -= scalar } -> std::same_as<T &>;

  { a.exp() } -> std::same_as<T>;
  { a.log() } -> std::same_as<T>;
  { a.sqrt() } -> std::same_as<T>;

  { a.dot(b) } -> std::same_as<double>;
  { a.hadamard(b) } -> std::same_as<T>;

  { a.sum() } -> std::same_as<double>;
  { a.size() } -> std::same_as<size_t>;

  { to_json(j, a) } -> std::same_as<void>;
  { from_json(j, a) } -> std::same_as<void>;

  {
    a.elementwise([](auto &&x) -> decltype(auto) { return x; })
  } -> std::same_as<T>;
};

#endif // ML_ALGEBRAIC_H
