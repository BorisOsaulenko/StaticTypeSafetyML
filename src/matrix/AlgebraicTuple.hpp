#ifndef ML_ALGEBRAICTUPLE_H
#define ML_ALGEBRAICTUPLE_H

#include <cstddef>
#include <nlohmann/json.hpp>
#include <tuple>
#include <utility>

// AlgebraicTuple: heterogeneous tuple of Algebraic types. Elementwise
// operations.
template <typename... Types>
  requires(sizeof...(Types) > 0)
class AlgebraicTuple {
  std::tuple<Types...> data_;
  using json = nlohmann::json;

public:
  using value_type = double;

  AlgebraicTuple() = default;
  explicit AlgebraicTuple(std::tuple<Types...> d) : data_(std::move(d)) {}

  // Construct from elements
  explicit AlgebraicTuple(const Types &...elems) : data_(elems...) {}

  static AlgebraicTuple zeros() {
    return AlgebraicTuple(std::make_tuple(Types::zeros()...));
  }

  static AlgebraicTuple random() {
    return AlgebraicTuple(std::make_tuple(Types::random()...));
  }

  template <size_t I> [[nodiscard]] const auto &get() const {
    return std::get<I>(data_);
  }

  template <size_t I> [[nodiscard]] auto &get() { return std::get<I>(data_); }

  [[nodiscard]] static constexpr size_t size() { return sizeof...(Types); }

  // Elementwise operations with another tuple
  [[nodiscard]] AlgebraicTuple operator+(const AlgebraicTuple &rhs) const {
    return binary_op(rhs, [](auto const &a, auto const &b) { return a + b; });
  }

  AlgebraicTuple &operator+=(const AlgebraicTuple &rhs) {
    binary_op_inplace(rhs, [](auto &a, auto const &b) { a += b; });
    return *this;
  }

  [[nodiscard]] AlgebraicTuple operator-(const AlgebraicTuple &rhs) const {
    return binary_op(rhs, [](auto const &a, auto const &b) { return a - b; });
  }

  AlgebraicTuple &operator-=(const AlgebraicTuple &rhs) {
    binary_op_inplace(rhs, [](auto &a, auto const &b) { a -= b; });
    return *this;
  }

  [[nodiscard]] AlgebraicTuple operator*(const AlgebraicTuple &rhs) const {
    return binary_op(rhs, [](auto const &a, auto const &b) { return a * b; });
  }

  AlgebraicTuple &operator*=(const AlgebraicTuple &rhs) {
    binary_op_inplace(rhs, [](auto &a, auto const &b) { a *= b; });
    return *this;
  }

  [[nodiscard]] AlgebraicTuple operator/(const AlgebraicTuple &rhs) const {
    return binary_op(rhs, [](auto const &a, auto const &b) { return a / b; });
  }

  AlgebraicTuple &operator/=(const AlgebraicTuple &rhs) {
    binary_op_inplace(rhs, [](auto &a, auto const &b) { a /= b; });
    return *this;
  }

  // Operations with scalar (double)
  [[nodiscard]] AlgebraicTuple operator+(const value_type scalar) const {
    return scalar_unary_op([&](auto const &a) { return a + scalar; });
  }

  AlgebraicTuple &operator+=(const value_type scalar) {
    scalar_unary_op_inplace([&](auto &a) { a += scalar; });
    return *this;
  }

  [[nodiscard]] AlgebraicTuple operator-(const value_type scalar) const {
    return scalar_unary_op([&](auto const &a) { return a - scalar; });
  }

  AlgebraicTuple &operator-=(const value_type scalar) {
    scalar_unary_op_inplace([&](auto &a) { a -= scalar; });
    return *this;
  }

  [[nodiscard]] AlgebraicTuple operator*(const value_type scalar) const {
    return scalar_unary_op([&](auto const &a) { return a * scalar; });
  }

  AlgebraicTuple &operator*=(const value_type scalar) {
    scalar_unary_op_inplace([&](auto &a) { a *= scalar; });
    return *this;
  }

  [[nodiscard]] AlgebraicTuple operator/(const value_type scalar) const {
    return scalar_unary_op([&](auto const &a) { return a / scalar; });
  }

  AlgebraicTuple &operator/=(const value_type scalar) {
    scalar_unary_op_inplace([&](auto &a) { a /= scalar; });
    return *this;
  }

  // Elementwise special ops
  [[nodiscard]] AlgebraicTuple log() const {
    return unary_op([](auto const &a) { return a.log(); });
  }

  [[nodiscard]] AlgebraicTuple exp() const {
    return unary_op([](auto const &a) { return a.exp(); });
  }

  [[nodiscard]] AlgebraicTuple sqrt() const {
    return unary_op([](auto const &a) { return a.sqrt(); });
  }

  [[nodiscard]] value_type dot(const AlgebraicTuple &rhs) const {
    return dot_impl(rhs, std::make_index_sequence<sizeof...(Types)>());
  }

  template <size_t... I>
  value_type dot_impl(const AlgebraicTuple &rhs,
                      std::index_sequence<I...>) const {
    return (value_type{0} + ... +
            std::get<I>(data_).dot(std::get<I>(rhs.data_)));
  }

  [[nodiscard]] AlgebraicTuple hadamard(const AlgebraicTuple &rhs) const {
    return binary_op(
        rhs, [](auto const &a, auto const &b) { return a.hadamard(b); });
  }

  [[nodiscard]] value_type sum() const {
    return unary_reduce([](auto const &a) { return a.sum(); });
  }

  template <typename F> AlgebraicTuple elementwise(F &&f) const {
    return unary_op(std::forward<F>(f));
  }

  template <size_t... Is>
  friend void to_json(json &j, const AlgebraicTuple &tuple,
                      std::index_sequence<Is...>) {
    j = json::array({std::get<Is>(tuple.data_)...});
  }

  friend void to_json(json &j, const AlgebraicTuple &tuple) {
    to_json(j, tuple, std::make_index_sequence<sizeof...(Types)>{});
  }

  template <size_t... Is>
  friend void from_json(const json &j, AlgebraicTuple &tuple,
                        std::index_sequence<Is...>) {
    tuple.data_ = std::make_tuple(j.at(Is).get<Types>()...);
  }

  friend void from_json(const json &j, AlgebraicTuple &tuple) {
    if (!j.is_array() || j.size() != sizeof...(Types))
      throw std::runtime_error("Invalid JSON for AlgebraicTuple");

    from_json(j, tuple, std::make_index_sequence<sizeof...(Types)>{});
  }

private:
  // Helpers
  template <typename F, size_t... Is>
  static AlgebraicTuple map_impl(const std::tuple<Types...> &t, F &&f,
                                 std::index_sequence<Is...>) {
    return AlgebraicTuple(std::make_tuple(f(std::get<Is>(t))...));
  }

  template <typename F, size_t... Is>
  static AlgebraicTuple map2_impl(const std::tuple<Types...> &a,
                                  const std::tuple<Types...> &b, F &&f,
                                  std::index_sequence<Is...>) {
    return AlgebraicTuple(
        std::make_tuple(f(std::get<Is>(a), std::get<Is>(b))...));
  }

  template <typename F> [[nodiscard]] AlgebraicTuple unary_op(F &&f) const {
    return map_impl(data_, std::forward<F>(f),
                    std::make_index_sequence<sizeof...(Types)>());
  }

  template <typename F>
  [[nodiscard]] AlgebraicTuple scalar_unary_op(F &&f) const {
    return map_impl(data_, std::forward<F>(f),
                    std::make_index_sequence<sizeof...(Types)>());
  }

  template <typename F, size_t... I>
  void scalar_unary_op_inplace_impl(F &&f, std::index_sequence<I...>) {
    (f(std::get<I>(data_)), ...);
  }

  template <typename F> void scalar_unary_op_inplace(F &&f) {
    scalar_unary_op_inplace_impl(std::forward<F>(f),
                                 std::make_index_sequence<sizeof...(Types)>());
  }

  template <typename F>
  [[nodiscard]] AlgebraicTuple binary_op(const AlgebraicTuple &rhs,
                                         F &&f) const {
    return map2_impl(data_, rhs.data_, std::forward<F>(f),
                     std::make_index_sequence<sizeof...(Types)>());
  }

  template <typename F, size_t... I>
  void binary_op_inplace_impl(const AlgebraicTuple &rhs, F &&f,
                              std::index_sequence<I...>) {
    (f(std::get<I>(data_), std::get<I>(rhs.data_)), ...);
  }

  template <typename F>
  void binary_op_inplace(const AlgebraicTuple &rhs, F &&f) {
    binary_op_inplace_impl(rhs, std::forward<F>(f),
                           std::make_index_sequence<sizeof...(Types)>());
  }

  template <typename F> value_type unary_reduce(F &&f) const {
    value_type total = 0;
    apply_for_each([&](auto const &elem) { total += f(elem); });
    return total;
  }

  // Apply helper: iterate over each element and call lambda(element)
  template <typename F, size_t... I>
  void apply_for_each_impl(F &&f, std::index_sequence<I...>) const {
    (f(std::get<I>(data_)), ...);
  }

  template <typename F> void apply_for_each(F &&f) const {
    apply_for_each_impl(std::forward<F>(f),
                        std::make_index_sequence<sizeof...(Types)>());
  }
};

#endif // ML_ALGEBRAICTUPLE_H
