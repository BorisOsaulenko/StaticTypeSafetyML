//
// Created by Boris Osaulenko on 19/07/2026.
//

#ifndef ML_ALGEBRAICARRAY_H
#define ML_ALGEBRAICARRAY_H
#include "Algebraic.h"
#include "Matrix.h"

#include <vector>

template <std::size_t Size, typename Type>
  requires Algebraic<std::remove_const_t<Type>>
class AlgebraicArray {
  std::vector<Type> data;
  using json = nlohmann::json;
  using RawType = std::remove_const_t<Type>;

public:
  using value_type = double;

  static AlgebraicArray zeros() {
    AlgebraicArray result;
    result.data.resize(Size, Type::zeros());
    return result;
  }

  static AlgebraicArray random() {
    AlgebraicArray result;
    result.data.resize(Size, Type::random());
    return result;
  }

  AlgebraicArray() { data.resize(Size); }

  explicit AlgebraicArray(const std::vector<Type> &vec) {
    assert(vec.size() == Size);
    data = vec;
  }

  explicit AlgebraicArray(const Type &default_val) {
    for (std::size_t i = 0; i < Size; ++i) {
      data[i] = default_val;
    }
  }

  AlgebraicArray(const std::initializer_list<Type> lst) {
    assert(lst.size() == Size);

    std::size_t idx = 0;
    for (const auto &value : lst)
      data[idx++] = value;
  }

  [[nodiscard]] const Type &operator[](std::size_t index) const {
    assert(index < Size);
    return data[index];
  }

  [[nodiscard]] Type &operator[](std::size_t index) {
    assert(index < Size);
    return data[index];
  }

  [[nodiscard]] static std::size_t size() { return Size; }

  [[nodiscard]] AlgebraicArray operator+(const value_type &scalar) const {
    return unary([scalar](const Type &a) { return a + scalar; });
  }

  [[nodiscard]] AlgebraicArray operator-(const value_type &scalar) const {
    return unary([scalar](const Type &a) { return a - scalar; });
  }

  [[nodiscard]] AlgebraicArray operator*(const value_type &scalar) const {
    return unary([scalar](const Type &a) { return a * scalar; });
  }

  [[nodiscard]] AlgebraicArray operator/(const value_type &scalar) const {
    return unary([scalar](const Type &a) { return a / scalar; });
  }

  AlgebraicArray &operator+=(const value_type &scalar) {
    return unary_inplace([scalar](Type &a) { a += scalar; });
  }

  AlgebraicArray &operator-=(const value_type &scalar) {
    return unary_inplace([scalar](Type &a) { a -= scalar; });
  }

  AlgebraicArray &operator*=(const value_type &scalar) {
    return unary_inplace([scalar](Type &a) { a *= scalar; });
  }

  AlgebraicArray &operator/=(const value_type &scalar) {
    return unary_inplace([scalar](Type &a) { a /= scalar; });
  }

  [[nodiscard]] AlgebraicArray operator+(const AlgebraicArray &rhs) const {
    return binary(rhs, [](const Type &a, const Type &b) { return a + b; });
  }

  [[nodiscard]] AlgebraicArray operator-(const AlgebraicArray &rhs) const {
    return binary(rhs, [](const Type &a, const Type &b) { return a - b; });
  }

  [[nodiscard]] AlgebraicArray operator*(const Type &rhs) const {
    return binary(rhs, [](const Type &a, const Type &b) { return a * b; });
  }

  [[nodiscard]] AlgebraicArray operator/(const AlgebraicArray &rhs) const {
    return binary(rhs, [](const Type &a, const Type &b) { return a / b; });
  }

  AlgebraicArray &operator+=(const AlgebraicArray &rhs) {
    return binary_inplace(rhs, [](Type &a, const Type &b) { a += b; });
  }

  AlgebraicArray &operator-=(const AlgebraicArray &rhs) {
    return binary_inplace(rhs, [](Type &a, const Type &b) { a -= b; });
  }

  AlgebraicArray &operator/=(const AlgebraicArray &rhs) {
    return binary_inplace(rhs, [](Type &a, const Type &b) { a /= b; });
  }

  [[nodiscard]] AlgebraicArray log() const {
    return unary([](const Type &e) { return e.log(); });
  }
  [[nodiscard]] AlgebraicArray exp() const {
    return unary([](const Type &e) { return e.exp(); });
  }
  [[nodiscard]] AlgebraicArray sqrt() const {
    return unary([](const Type &e) { return e.sqrt(); });
  }

  [[nodiscard]] value_type dot(const AlgebraicArray &rhs) const {
    value_type total = 0;
    for (std::size_t i = 0; i < Size; ++i) {
      total += data[i].dot(rhs.data[i]);
    }
    return total;
  }

  [[nodiscard]] AlgebraicArray hadamard(const AlgebraicArray &rhs) const {
    return binary(rhs,
                  [](const Type &a, const Type &b) { return a.hadamard(b); });
  }

  [[nodiscard]] value_type sum() const {
    value_type total = 0;
    for (const auto &value : data) {
      total += value.sum();
    }
    return total;
  }

  template <typename F> AlgebraicArray elementwise(F &&f) const {
    AlgebraicArray result;
    for (auto &i : data) {
      result.data.push_back(i.elementwise(f));
    }
    return result;
  }

  Type *get_data() noexcept { return data.data(); }
  const Type *get_data() const noexcept { return data.data(); }

  friend void to_json(json &j, const AlgebraicArray &arr) { j = arr.data; }

  friend void from_json(const json &j, AlgebraicArray &arr) { j.get_to(arr); }

private:
  template <typename F> AlgebraicArray &unary_inplace(F &&func) {
    for (std::size_t i = 0; i < Size; ++i)
      func(data[i]);

    return *this;
  }

  template <typename F>
  [[nodiscard]]
  AlgebraicArray unary(F &&func) const {
    AlgebraicArray result;
    result.data.resize(Size);

    for (std::size_t i = 0; i < Size; ++i)
      result.data[i] = func(data[i]);

    return result;
  }

  template <typename F>
  AlgebraicArray &binary_inplace(const AlgebraicArray &rhs, F &&func) {
    for (std::size_t i = 0; i < Size; ++i)
      func(data[i], rhs.data[i]);

    return *this;
  }

  template <typename F>
  [[nodiscard]]
  AlgebraicArray binary(const AlgebraicArray &rhs, F &&func) const {
    AlgebraicArray result;
    result.data.resize(Size);

    for (std::size_t i = 0; i < Size; ++i)
      result.data[i] = func(data[i], rhs.data[i]);

    return result;
  }
};

static_assert(Algebraic<AlgebraicArray<4, Matrix<3, 4>>>);

#endif // ML_ALGEBRAICARRAY_H
