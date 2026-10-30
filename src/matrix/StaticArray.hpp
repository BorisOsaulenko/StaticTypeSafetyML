#ifndef ML_ALGEBRAICARRAY_H
#define ML_ALGEBRAICARRAY_H
#include "Algebraic.hpp"

#include <vector>

template <size_t SIZE, typename Type> class StaticArray {
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
    for (size_t i = 0; i < SIZE; ++i) {
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
    for (size_t i = 0; i < SIZE; ++i) {
      data[i] = default_val;
    }
  }

  StaticArray(const std::initializer_list<Type> lst) {
    assert(lst.size() == SIZE);
    data.resize(SIZE);

    size_t idx = 0;
    for (const auto &value : lst)
      data[idx++] = value;
  }

  StaticArray(const StaticArray &other) {
    data.resize(SIZE);
    size_t idx = 0;
    for (const auto &value : other.data)
      data[idx++] = value;
  }

  StaticArray &operator=(const StaticArray &other) {
    data.resize(SIZE);
    for (size_t idx = 0; idx < SIZE; ++idx)
      data[idx] = other.data[idx];
    return *this;
  }

  [[nodiscard]] const Type &operator[](size_t index) const {
    assert(index < SIZE);
    return data[index];
  }

  [[nodiscard]] Type &operator[](size_t index) {
    assert(index < SIZE);
    return data[index];
  }

  [[nodiscard]] static size_t size() { return SIZE; }

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
    for (size_t i = 0; i < SIZE; ++i) {
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
    for (size_t i = 0; i < SIZE; ++i) {
      result.data[i] = data[i].elementwise(f);
    }
    return result;
  }

  operator std::span<RawType, SIZE>() noexcept {
    return std::span<RawType, SIZE>(data.data(), SIZE);
  }
  operator std::span<const RawType, SIZE>() const noexcept {
    return std::span<const RawType, SIZE>(data.data(), SIZE);
  }

  Type *get_data() noexcept { return data.data(); }
  const Type *get_data() const noexcept { return data.data(); }

  friend void to_json(json &j, const StaticArray &arr) { j = arr.data; }

  friend void from_json(const json &j, StaticArray &arr) { j.get_to(arr.data); }

private:
  template <typename F> StaticArray &unary_inplace(F &&func) {
    for (size_t i = 0; i < SIZE; ++i)
      func(data[i]);

    return *this;
  }

  template <typename F>
  [[nodiscard]]
  StaticArray unary(F &&func) const {
    StaticArray result;
    result.data.resize(SIZE);

    for (size_t i = 0; i < SIZE; ++i)
      result.data[i] = func(data[i]);

    return result;
  }

  template <typename F>
  StaticArray &binary_inplace(const StaticArray &rhs, F &&func) {
    for (size_t i = 0; i < SIZE; ++i)
      func(data[i], rhs.data[i]);

    return *this;
  }

  template <typename F>
  [[nodiscard]]
  StaticArray binary(const StaticArray &rhs, F &&func) const {
    StaticArray result;
    result.data.resize(SIZE);

    for (size_t i = 0; i < SIZE; ++i)
      result.data[i] = func(data[i], rhs.data[i]);

    return result;
  }
};

#endif // ML_ALGEBRAICARRAY_H