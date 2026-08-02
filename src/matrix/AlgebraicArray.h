//
// Created by Boris Osaulenko on 19/07/2026.
//

#ifndef ML_ALGEBRAICARRAY_H
#define ML_ALGEBRAICARRAY_H
#include "Algebraic.h"
#include "Matrix.h"

#include <vector>

template <std::size_t SIZE, typename Type>
  requires Algebraic<std::remove_const_t<Type>>
class AlgebraicArray {
  std::vector<Type> data;
  using json = nlohmann::json;
  using RawType = std::remove_const_t<Type>;

public:
  using value_type = double;

  static AlgebraicArray zeros() {
    AlgebraicArray result;
    result.data.resize(SIZE, Type::zeros());
    return result;
  }

  static AlgebraicArray random() {
    AlgebraicArray result;
    for (std::size_t i = 0; i < SIZE; ++i) {
      result.data[i] = Type::random();
    }
    return result;
  }

  AlgebraicArray() { data.resize(SIZE); }

  explicit AlgebraicArray(const std::vector<Type> &vec) {
    assert(vec.size() == SIZE);
    data = vec;
  }

  explicit AlgebraicArray(const Type &default_val) {
    data.resize(SIZE);
    for (std::size_t i = 0; i < SIZE; ++i) {
      data[i] = default_val;
    }
  }

  AlgebraicArray(const std::initializer_list<Type> lst) {
    assert(lst.size() == SIZE);
    data.resize(SIZE);

    std::size_t idx = 0;
    for (const auto &value : lst)
      data[idx++] = value;
  }

  AlgebraicArray(const AlgebraicArray &other) {
    data.resize(SIZE);
    std::size_t idx = 0;
    for (const auto &value : other.data)
      data[idx++] = value;
  }

  AlgebraicArray &operator=(const AlgebraicArray &other) {
    data.resize(SIZE);
    for (std::size_t idx = 0; idx < SIZE; ++idx)
      data[idx] = other.data[idx];
    return *this;
  }

  [[nodiscard]] const Type &operator[](std::size_t index) const {
    assert(index < SIZE);
    return data[index];
  }

  [[nodiscard]] Type &operator[](std::size_t index) {
    assert(index < SIZE);
    return data[index];
  }

  [[nodiscard]] static std::size_t size() { return SIZE; }

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
    for (std::size_t i = 0; i < SIZE; ++i) {
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
    for (std::size_t i = 0; i < SIZE; ++i) {
      result.data[i] = data[i].elementwise(f);
    }
    return result;
  }

  Type *get_data() noexcept { return data.data(); }
  const Type *get_data() const noexcept { return data.data(); }

  friend void to_json(json &j, const AlgebraicArray &arr) { j = arr.data; }

  friend void from_json(const json &j, AlgebraicArray &arr) {
    j.get_to(arr.data);
  }

private:
  template <typename F> AlgebraicArray &unary_inplace(F &&func) {
    for (std::size_t i = 0; i < SIZE; ++i)
      func(data[i]);

    return *this;
  }

  template <typename F>
  [[nodiscard]]
  AlgebraicArray unary(F &&func) const {
    AlgebraicArray result;
    result.data.resize(SIZE);

    for (std::size_t i = 0; i < SIZE; ++i)
      result.data[i] = func(data[i]);

    return result;
  }

  template <typename F>
  AlgebraicArray &binary_inplace(const AlgebraicArray &rhs, F &&func) {
    for (std::size_t i = 0; i < SIZE; ++i)
      func(data[i], rhs.data[i]);

    return *this;
  }

  template <typename F>
  [[nodiscard]]
  AlgebraicArray binary(const AlgebraicArray &rhs, F &&func) const {
    AlgebraicArray result;
    result.data.resize(SIZE);

    for (std::size_t i = 0; i < SIZE; ++i)
      result.data[i] = func(data[i], rhs.data[i]);

    return result;
  }
};

template <std::size_t SIZE, typename T>
  requires Algebraic<std::remove_const_t<T>>
class AlgebraicSpan {
public:
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using size_type = std::size_t;
  using RawT = std::remove_const_t<T>;

  constexpr AlgebraicSpan() = default;

  constexpr AlgebraicSpan(const pointer begin, const pointer end)
      : begin_(begin), end_(end) {
    assert(static_cast<size_type>(end - begin) == SIZE);
  }

  constexpr explicit AlgebraicSpan(const pointer begin)
      : begin_(begin), end_(begin + SIZE) {}

  AlgebraicSpan(AlgebraicArray<SIZE, RawT> &arr)
      : begin_(arr.get_data()), end_(arr.get_data() + SIZE) {}
  AlgebraicSpan(const AlgebraicArray<SIZE, RawT> &arr)
      : begin_(arr.get_data()), end_(arr.get_data() + SIZE) {}
  AlgebraicSpan(AlgebraicSpan<SIZE, RawT> &other)
      : begin_(other.begin_), end_(other.end_) {}
  AlgebraicSpan(const AlgebraicSpan<SIZE, RawT> &other)
      : begin_(other.begin_), end_(other.end_) {}

  [[nodiscard]] constexpr reference operator[](size_type i) const {
    assert(i < SIZE);
    return begin_[i];
  }

  template <std::size_t FIRST, std::size_t LAST>
  [[nodiscard]] constexpr AlgebraicSpan<LAST - FIRST, T> subspan() const {
    static_assert(FIRST <= LAST, "FIRST must be <= LAST");
    static_assert(LAST <= SIZE, "LAST must be <= SIZE");

    return AlgebraicSpan<LAST - FIRST, T>(begin_ + FIRST);
  }

  [[nodiscard]] AlgebraicArray<SIZE, RawT>
  hadamard(const AlgebraicSpan<SIZE, const RawT> &rhs) const {
    AlgebraicArray<SIZE, RawT> result;
    for (std::size_t i = 0; i < SIZE; ++i) {
      result[i] = (*this)[i].hadamard(rhs[i]);
    }
    return result;
  }

  [[nodiscard]] constexpr pointer data() const noexcept { return begin_; }

  [[nodiscard]] constexpr pointer begin() const noexcept { return begin_; }

  [[nodiscard]] constexpr pointer end() const noexcept { return end_; }

  [[nodiscard]] static constexpr size_type size() noexcept { return SIZE; }

private:
  pointer begin_;
  pointer end_;
};

template <std::size_t SIZE, typename RawT>
AlgebraicSpan(const AlgebraicArray<SIZE, RawT> &)
    -> AlgebraicSpan<SIZE, const RawT>;

static_assert(Algebraic<AlgebraicArray<4, Matrix<3, 4>>>);

#endif // ML_ALGEBRAICSPAN_H
