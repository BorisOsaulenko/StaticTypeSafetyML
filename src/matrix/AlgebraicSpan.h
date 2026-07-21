//
// Created by Boris Osaulenko on 20/07/2026.
//

#ifndef ML_ALGEBRAICSPAN_H
#define ML_ALGEBRAICSPAN_H
#include "AlgebraicArray.h"

#include <cassert>

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
  //
  // template <typename U>
  //   requires std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>
  // constexpr AlgebraicSpan(const AlgebraicArray<SIZE, U> &arr)
  //     : begin_(arr.get_data()), end_(arr.get_data() + SIZE) {}

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

  [[nodiscard]] constexpr pointer data() const noexcept { return begin_; }

  [[nodiscard]] constexpr pointer begin() const noexcept { return begin_; }

  [[nodiscard]] constexpr pointer end() const noexcept { return end_; }

  [[nodiscard]] static constexpr size_type size() noexcept { return SIZE; }

private:
  pointer begin_;
  pointer end_;
};

#endif // ML_ALGEBRAICSPAN_H
