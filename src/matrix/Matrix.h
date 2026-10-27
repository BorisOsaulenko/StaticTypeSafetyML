#ifndef ML_MATRIX_H
#define ML_MATRIX_H

#include "Algebraic.h"

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <ostream>
#include <random>

template <class Derived, size_t R, size_t C> class MatrixBase {
public:
  using value_type = double;
  using storage_type = std::vector<value_type>;
  using json = nlohmann::json;
  static constexpr size_t SIZE = R * C;

  constexpr MatrixBase() : data_(SIZE, 0) {}
  constexpr explicit MatrixBase(const value_type fill) : data_(R * C, fill) {}
  static constexpr MatrixBase zeros() { return MatrixBase(0); }
  static constexpr MatrixBase random() {
    MatrixBase result;
    for (size_t idx = 0; idx < SIZE; idx++)
      result.data_[idx] = dist(rng);
    return result;
  }

protected:
  constexpr Derived &derived() { return static_cast<Derived &>(*this); }

  [[nodiscard]] constexpr const Derived &derived() const {
    return static_cast<const Derived &>(*this);
  }

public:
  constexpr value_type operator[](const size_t &index) const {
    return data_[index];
  }
  value_type &operator[](const size_t &index) { return data_[index]; }

  constexpr value_type operator[](const size_t &r, const size_t &c) const {
    return data_[r * C + c];
  }
  value_type &operator[](const size_t &r, const size_t &c) {
    return data_[r * C + c];
  }

  static void set_seed(const size_t &seed) { rng.seed(seed); }

  [[nodiscard]] Derived sqrt() const {
    Derived result;
    for (size_t i = 0; i < data_.size(); ++i)
      result.storage()[i] = std::sqrt(data_[i]);
    return result;
  }

  [[nodiscard]] Derived log() const {
    Derived result;
    for (size_t i = 0; i < data_.size(); ++i)
      result.storage()[i] = std::log(data_[i]);
    return result;
  }

  [[nodiscard]] Derived exp() const {
    Derived result;
    for (size_t i = 0; i < data_.size(); ++i)
      result.storage()[i] = std::exp(data_[i]);
    return result;
  }

  [[nodiscard]] Derived operator+(const Derived &rhs) const {
    Derived result;
    for (size_t i = 0; i < data_.size(); ++i)
      result.storage()[i] = data_[i] + rhs.storage()[i];
    return result;
  }

  Derived &operator+=(const Derived &rhs) {
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] += rhs.storage()[i];
    return derived();
  }

  [[nodiscard]] Derived operator-(const Derived &rhs) const {
    Derived result;
    for (size_t i = 0; i < data_.size(); ++i)
      result.storage()[i] = data_[i] - rhs.storage()[i];
    return result;
  }

  Derived &operator-=(const Derived &rhs) {
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] -= rhs.storage()[i];
    return derived();
  }

  [[nodiscard]] Derived operator/(const Derived &rhs) const {
    Derived result;
    for (size_t i = 0; i < data_.size(); ++i)
      result.storage()[i] = data_[i] / rhs.storage()[i];
    return result;
  }

  Derived &operator/=(const Derived &rhs) {
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] /= rhs.storage()[i];
    return derived();
  }

  [[nodiscard]] Derived operator*(const value_type scalar) const {
    Derived result;
    for (size_t i = 0; i < data_.size(); ++i)
      result.storage()[i] = data_[i] * scalar;
    return result;
  }

  Derived &operator*=(const value_type scalar) {
    for (double &i : data_)
      i *= scalar;
    return derived();
  }

  [[nodiscard]] Derived operator/(const value_type scalar) const {
    Derived result;
    for (size_t i = 0; i < data_.size(); ++i)
      result.storage()[i] = data_[i] / scalar;
    return result;
  }

  Derived &operator/=(const value_type scalar) {
    for (double &i : data_)
      i /= scalar;
    return derived();
  }

  [[nodiscard]] Derived operator+(const value_type scalar) const {
    Derived result;
    for (size_t i = 0; i < data_.size(); ++i)
      result.storage()[i] = data_[i] + scalar;
    return result;
  }

  Derived &operator+=(const value_type scalar) {
    for (double &i : data_)
      i += scalar;
    return derived();
  }

  [[nodiscard]] Derived operator-(const value_type scalar) const {
    Derived result;
    for (size_t i = 0; i < data_.size(); ++i)
      result.storage()[i] = data_[i] - scalar;
    return result;
  }

  Derived &operator-=(const value_type scalar) {
    for (double &i : data_)
      i -= scalar;
    return derived();
  }

  MatrixBase &operator=(const Derived &rhs) {
    data_.resize(SIZE);
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] = rhs.storage()[i];
    return *this;
  }

  [[nodiscard]] Derived hadamard(const Derived &rhs) const {
    Derived result;
    for (size_t i = 0; i < data_.size(); ++i)
      result.storage()[i] = data_[i] * rhs.storage()[i];
    return result;
  }

  [[nodiscard]] double dot(const Derived &rhs) const {
    double result = 0;
    for (size_t i = 0; i < data_.size(); ++i)
      result += data_[i] * rhs.storage()[i];
    return result;
  }

  [[nodiscard]] double sum() const {
    double result = 0;
    for (const double i : data_)
      result += i;
    return result;
  }

  [[nodiscard]] double max() const {
    double result = 0;
    for (const double i : data_)
      result = std::max(result, i);
    return result;
  }

  [[nodiscard]] bool operator==(MatrixBase other) const {
    for (size_t i = 0; i < data_.size(); i++) {
      if (data_[i] != other[i])
        return false;
    }
    return true;
  }

  [[nodiscard]] value_type *data() { return data_.data(); }
  [[nodiscard]] const value_type *data() const { return data_.data(); }

  template <typename F> Derived elementwise(F &&f) const {
    Derived result;
    for (size_t i = 0; i < SIZE; ++i) {
      result.storage()[i] = f(data_[i]);
    }
    return result;
  }

  friend void to_json(json &j, const Derived &d) { j = d.data_; }
  friend void from_json(const json &j, Derived &d) { j.get_to(d.data_); }

  [[nodiscard]] size_t size() const { return data_.size(); }

  storage_type &storage() { return data_; }
  [[nodiscard]] const storage_type &storage() const { return data_; }

protected:
  storage_type data_;

private:
  static inline std::mt19937 rng{std::random_device{}()};
  static inline std::uniform_real_distribution<> dist{-0.5, 0.5};
};

template <size_t R> class Vector;

template <size_t R, size_t C>
class Matrix : public MatrixBase<Matrix<R, C>, R, C> {
public:
  using Base = MatrixBase<Matrix, R, C>;
  using Base::Base; // inherit constructors
  using value_type = Base::value_type;
  using Base::operator*;
  using Base::operator*=;

  Matrix() = default;

  Matrix(std::initializer_list<value_type> init_list) {
    assert(init_list.size() == R * C);
    size_t i = 0;
    for (const auto &elem : init_list) {
      this->data_[i++] = elem;
    }
  }

  Matrix(std::initializer_list<std::initializer_list<value_type>> rows) {
    assert(rows.size() == R);
    size_t r = 0;
    for (const auto &row : rows) {
      assert(row.size() == C);
      size_t c = 0;
      for (const auto &elem : row)
        this->data_[r * C + c++] = elem;
      ++r;
    }
  }

  Matrix(MatrixBase<Matrix, R, C> const &base) {
    for (size_t row = 0; row < R; ++row) {
      for (size_t col = 0; col < C; ++col) {
        this->data_[row * C + col] = base[row * C + col];
      }
    }
  }

  static Matrix outer_product(const Vector<R> &lhs, const Vector<C> &rhs);

  value_type &operator[](const size_t row, const size_t col) {
    return this->data_[row * C + col];
  }

  const value_type &operator[](const size_t row, const size_t col) const {
    return this->data_[row * C + col];
  }

  void set_row(size_t row, const Vector<C> &vec) {
    for (size_t col = 0; col < C; ++col) {
      this->data_[row * C + col] = vec[col];
    }
  }

  void set_col(size_t col, const Vector<R> &vec) {
    for (size_t row = 0; row < R; ++row) {
      this->data_[row * C + col] = vec[row];
    }
  }

  [[nodiscard]] Vector<R> get_col(const size_t col) const {
    Vector<R> vec;
    for (size_t row = 0; row < R; ++row) {
      vec[row] = this->data_[row * C + col];
    }
    return vec;
  }

  [[nodiscard]] Vector<C> get_row(const size_t row) const {
    Vector<C> vec;
    for (size_t col = 0; col < C; ++col) {
      vec[col] = this->data_[row * C + col];
    }
    return vec;
  }

  template <size_t KR, size_t KC>
    requires(KR > 0 && KC > 0 && R >= KR && C >= KC)
  [[nodiscard]] constexpr Matrix<R - KR + 1, C - KC + 1>
  valid_cross_correlation(const Matrix<KR, KC> &kernel) const {
    Matrix<R - KR + 1, C - KC + 1> result;

    for (size_t row = 0; row < R - KR + 1; ++row) {
      for (size_t col = 0; col < C - KC + 1; ++col) {
        typename Base::value_type sum = 0;

        for (size_t kr = 0; kr < KR; ++kr) {
          for (size_t kc = 0; kc < KC; ++kc) {
            sum += (*this)[row + kr, col + kc] * kernel[kr, kc];
          }
        }

        result[row, col] = sum;
      }
    }

    return result;
  }

  template <size_t KR, size_t KC>
    requires(KR > 0 && KC > 0)
  [[nodiscard]] constexpr Matrix<R + KR - 1, C + KC - 1>
  full_convolution(const Matrix<KR, KC> &kernel) const {
    constexpr size_t out_rows = R + KR - 1;
    constexpr size_t out_cols = C + KC - 1;

    Matrix<out_rows, out_cols> result;

    for (size_t out_row = 0; out_row < out_rows; ++out_row) {
      for (size_t out_col = 0; out_col < out_cols; ++out_col) {
        typename Base::value_type sum = 0;

        for (size_t kr = 0; kr < KR; ++kr) {
          for (size_t kc = 0; kc < KC; ++kc) {
            const long in_row =
                static_cast<size_t>(out_row) - static_cast<size_t>(kr);
            const long in_col =
                static_cast<size_t>(out_col) - static_cast<size_t>(kc);

            if (in_row >= 0 && static_cast<size_t>(in_row) < R && in_col >= 0 &&
                static_cast<size_t>(in_col) < C) {
              sum += (*this)[in_row, in_col] * kernel[kr, kc];
            }
          }
        }

        result[out_row, out_col] = sum;
      }
    }

    return result;
  }

  [[nodiscard]] constexpr Matrix<C, R> transpose() const {
    Matrix<C, R> result;

    for (size_t row = 0; row < R; ++row) {
      for (size_t col = 0; col < C; ++col) {
        result[col, row] = (*this)[row, col];
      }
    }

    return result;
  }

  friend std::ostream &operator<<(std::ostream &os, const Matrix<R, C> &obj) {
    for (size_t row = 0; row < R; ++row) {
      for (size_t col = 0; col < C; ++col) {
        os << obj[row, col] << " ";
      }
      os << std::endl;
    }
    return os;
  }
};

template <size_t R> class Vector : public MatrixBase<Vector<R>, R, 1> {
public:
  using Base = MatrixBase<Vector, R, 1>;
  using Base::Base; // inherit constructors
  using value_type = Base::value_type;
  using Base::operator*;
  using Base::operator*=;
  using Base::elementwise;

  Vector(std::initializer_list<value_type> init_list) {
    assert(init_list.size() == R);
    size_t i = 0;
    for (const auto &elem : init_list) {
      this->data_[i++] = elem;
    }
  }

  Vector(MatrixBase<Vector, R, 1> base) {
    for (size_t i = 0; i < R; ++i) {
      this->data_[i] = base[i];
    }
  }

  value_type &operator[](size_t i) { return this->data_[i]; }

  const value_type &operator[](size_t i) const { return this->data_[i]; }

  value_type &operator[](size_t row, size_t col) {
    assert(col == 0);
    return this->data_[row];
  }

  const value_type &operator[](size_t row, const size_t col) const {
    assert(col == 0);
    return this->data_[row];
  }
};

template <size_t R, size_t C>
[[nodiscard]]
Matrix<R, C> operator*(const double scalar, const Matrix<R, C> &rhs) {
  Matrix<R, C> result;
  for (size_t row = 0; row < R; ++row) {
    for (size_t col = 0; col < C; ++col) {
      result[row, col] = scalar * rhs[row, col];
    }
  }
  return result;
}

template <size_t R, size_t C, size_t OtherC>
[[nodiscard]]
Matrix<R, OtherC> operator*(const Matrix<R, C> &lhs,
                            const Matrix<C, OtherC> &rhs) {
  Matrix<R, OtherC> result;

  for (size_t row = 0; row < R; ++row) {
    for (size_t col = 0; col < OtherC; ++col) {
      typename Matrix<R, C>::value_type sum = 0;

      for (size_t k = 0; k < C; ++k)
        sum += lhs[row, k] * rhs[k, col];

      result[row, col] = sum;
    }
  }

  return result;
}

template <size_t R, size_t C>
[[nodiscard]]
Vector<R> operator*(const Matrix<R, C> &lhs, const Vector<C> &rhs) {
  Vector<R> result;

  for (size_t row = 0; row < R; ++row) {
    typename Matrix<R, C>::value_type sum = 0;

    for (size_t k = 0; k < C; ++k)
      sum += lhs[row, k] * rhs[k];

    result[row] = sum;
  }

  return result;
}

template <size_t R, size_t C>
Matrix<R, C> Matrix<R, C>::outer_product(const Vector<R> &lhs,
                                         const Vector<C> &rhs) {
  Matrix result;

  for (size_t row = 0; row < R; ++row) {
    for (size_t col = 0; col < C; ++col) {
      result[row, col] = lhs[row] * rhs[col];
    }
  }

  return result;
}

static_assert(Algebraic<Matrix<2, 3>>);
static_assert(Algebraic<Vector<3>>);

#endif // ML_MATRIX_H
