//
// Created by Boris Osaulenko on 19/07/2026.
//

#ifndef ML_ALGEBRAICARRAY_H
#define ML_ALGEBRAICARRAY_H
#include "Algebraic.h"

#include <vector>

template <std::size_t SIZE, typename Type> class StaticArray {
  std::vector<Type> data;
  using json = nlohmann::json;
  using RawType = std::remove_const_t<Type>;

public:
  using value_type = double;

  static StaticArray zeros()
    requires(Algebraic<RawType>)
  {
    StaticArray result;
    result.data.resize(SIZE, Type::zeros());
    return result;
  }

  static StaticArray random()
    requires(Algebraic<RawType>)
  {
    StaticArray result;
    for (std::size_t i = 0; i < SIZE; ++i) {
      result.data[i] = Type::random();
    }
    return result;
  }

  StaticArray() { data.resize(SIZE); }

  explicit StaticArray(const std::vector<Type> &vec) {
    assert(vec.size() == SIZE);
    data = vec;
  }

  explicit StaticArray(const Type &default_val) {
    data.resize(SIZE);
    for (std::size_t i = 0; i < SIZE; ++i) {
      data[i] = default_val;
    }
  }

  StaticArray(const std::initializer_list<Type> lst) {
    assert(lst.size() == SIZE);
    data.resize(SIZE);

    std::size_t idx = 0;
    for (const auto &value : lst)
      data[idx++] = value;
  }

  StaticArray(const StaticArray &other) {
    data.resize(SIZE);
    std::size_t idx = 0;
    for (const auto &value : other.data)
      data[idx++] = value;
  }

  StaticArray &operator=(const StaticArray &other) {
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

  [[nodiscard]] StaticArray operator+(const value_type &scalar) const
    requires(Algebraic<RawType>)
  {
    return unary([scalar](const Type &a) { return a + scalar; });
  }

  [[nodiscard]] StaticArray operator-(const value_type &scalar) const
    requires(Algebraic<RawType>)
  {
    return unary([scalar](const Type &a) { return a - scalar; });
  }

  [[nodiscard]] StaticArray operator*(const value_type &scalar) const
    requires(Algebraic<RawType>)
  {
    return unary([scalar](const Type &a) { return a * scalar; });
  }

  [[nodiscard]] StaticArray operator/(const value_type &scalar) const
    requires(Algebraic<RawType>)
  {
    return unary([scalar](const Type &a) { return a / scalar; });
  }

  StaticArray &operator+=(const value_type &scalar)
    requires(Algebraic<RawType>)
  {
    return unary_inplace([scalar](Type &a) { a += scalar; });
  }

  StaticArray &operator-=(const value_type &scalar)
    requires(Algebraic<RawType>)
  {
    return unary_inplace([scalar](Type &a) { a -= scalar; });
  }

  StaticArray &operator*=(const value_type &scalar)
    requires(Algebraic<RawType>)
  {
    return unary_inplace([scalar](Type &a) { a *= scalar; });
  }

  StaticArray &operator/=(const value_type &scalar)
    requires(Algebraic<RawType>)
  {
    return unary_inplace([scalar](Type &a) { a /= scalar; });
  }

  [[nodiscard]] StaticArray operator+(const StaticArray &rhs) const
    requires(Algebraic<RawType>)
  {
    return binary(rhs, [](const Type &a, const Type &b) { return a + b; });
  }

  [[nodiscard]] StaticArray operator-(const StaticArray &rhs) const
    requires(Algebraic<RawType>)
  {
    return binary(rhs, [](const Type &a, const Type &b) { return a - b; });
  }

  [[nodiscard]] StaticArray operator*(const Type &rhs) const
    requires(Algebraic<RawType>)
  {
    return binary(rhs, [](const Type &a, const Type &b) { return a * b; });
  }

  [[nodiscard]] StaticArray operator/(const StaticArray &rhs) const
    requires(Algebraic<RawType>)
  {
    return binary(rhs, [](const Type &a, const Type &b) { return a / b; });
  }

  StaticArray &operator+=(const StaticArray &rhs)
    requires(Algebraic<RawType>)
  {
    return binary_inplace(rhs, [](Type &a, const Type &b) { a += b; });
  }

  StaticArray &operator-=(const StaticArray &rhs)
    requires(Algebraic<RawType>)
  {
    return binary_inplace(rhs, [](Type &a, const Type &b) { a -= b; });
  }

  StaticArray &operator/=(const StaticArray &rhs)
    requires(Algebraic<RawType>)
  {
    return binary_inplace(rhs, [](Type &a, const Type &b) { a /= b; });
  }

  [[nodiscard]] StaticArray log() const
    requires(Algebraic<RawType>)
  {
    return unary([](const Type &e) { return e.log(); });
  }
  [[nodiscard]] StaticArray exp() const
    requires(Algebraic<RawType>)
  {
    return unary([](const Type &e) { return e.exp(); });
  }
  [[nodiscard]] StaticArray sqrt() const
    requires(Algebraic<RawType>)
  {
    return unary([](const Type &e) { return e.sqrt(); });
  }

  [[nodiscard]] value_type dot(const StaticArray &rhs) const
    requires(Algebraic<RawType>)
  {
    value_type total = 0;
    for (std::size_t i = 0; i < SIZE; ++i) {
      total += data[i].dot(rhs.data[i]);
    }
    return total;
  }

  [[nodiscard]] StaticArray hadamard(const StaticArray &rhs) const
    requires(Algebraic<RawType>)
  {
    return binary(rhs,
                  [](const Type &a, const Type &b) { return a.hadamard(b); });
  }

  [[nodiscard]] value_type sum() const
    requires(Algebraic<RawType>)
  {
    value_type total = 0;
    for (const auto &value : data) {
      total += value.sum();
    }
    return total;
  }

  template <typename F>
  StaticArray elementwise(F &&f) const
    requires(Algebraic<RawType>)
  {
    StaticArray result;
    for (std::size_t i = 0; i < SIZE; ++i) {
      result.data[i] = data[i].elementwise(f);
    }
    return result;
  }

  Type *get_data() noexcept { return data.data(); }
  const Type *get_data() const noexcept { return data.data(); }

  friend void to_json(json &j, const StaticArray &arr) { j = arr.data; }

  friend void from_json(const json &j, StaticArray &arr) { j.get_to(arr.data); }

private:
  template <typename F> StaticArray &unary_inplace(F &&func) {
    for (std::size_t i = 0; i < SIZE; ++i)
      func(data[i]);

    return *this;
  }

  template <typename F>
  [[nodiscard]]
  StaticArray unary(F &&func) const {
    StaticArray result;
    result.data.resize(SIZE);

    for (std::size_t i = 0; i < SIZE; ++i)
      result.data[i] = func(data[i]);

    return result;
  }

  template <typename F>
  StaticArray &binary_inplace(const StaticArray &rhs, F &&func) {
    for (std::size_t i = 0; i < SIZE; ++i)
      func(data[i], rhs.data[i]);

    return *this;
  }

  template <typename F>
  [[nodiscard]]
  StaticArray binary(const StaticArray &rhs, F &&func) const {
    StaticArray result;
    result.data.resize(SIZE);

    for (std::size_t i = 0; i < SIZE; ++i)
      result.data[i] = func(data[i], rhs.data[i]);

    return result;
  }
};

template <std::size_t SIZE, typename T> class StaticSpan {
public:
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using size_type = std::size_t;
  using RawType = std::remove_const_t<T>;

  constexpr StaticSpan() = default;

  constexpr StaticSpan(const pointer begin, const pointer end)
      : begin_(begin), end_(end) {
    assert(static_cast<size_type>(end - begin) == SIZE);
  }

  constexpr explicit StaticSpan(const pointer begin)
      : begin_(begin), end_(begin + SIZE) {}

  StaticSpan(StaticArray<SIZE, RawType> &arr)
      : begin_(arr.get_data()), end_(arr.get_data() + SIZE) {}
  StaticSpan(const StaticArray<SIZE, RawType> &arr)
      : begin_(arr.get_data()), end_(arr.get_data() + SIZE) {}
  StaticSpan(StaticSpan<SIZE, RawType> &other)
      : begin_(other.begin_), end_(other.end_) {}
  StaticSpan(const StaticSpan<SIZE, RawType> &other)
      : begin_(other.begin_), end_(other.end_) {}

  [[nodiscard]] constexpr reference operator[](size_type i) const {
    assert(i < SIZE);
    return begin_[i];
  }

  template <std::size_t FIRST, std::size_t LAST>
  [[nodiscard]] constexpr StaticSpan<LAST - FIRST, T> subspan() const {
    static_assert(FIRST <= LAST, "FIRST must be <= LAST");
    static_assert(LAST <= SIZE, "LAST must be <= SIZE");

    return StaticSpan<LAST - FIRST, T>(begin_ + FIRST);
  }

  [[nodiscard]] StaticArray<SIZE, RawType>
  hadamard(const StaticSpan<SIZE, const RawType> &rhs) const
    requires(Algebraic<RawType>)
  {
    StaticArray<SIZE, RawType> result;
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
StaticSpan(const StaticArray<SIZE, RawT> &) -> StaticSpan<SIZE, const RawT>;

#endif // ML_ALGEBRAICSPAN_H
