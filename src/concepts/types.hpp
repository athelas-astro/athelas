#pragma once

#include <cstddef>

namespace athelas {

template <typename F>
concept Functor = requires(F f) {
  { &F::operator() };
};

template <typename T>
concept VectorLike = requires(T v, std::size_t i) {
  // must support indexing
  { v[i] } -> std::same_as<typename T::value_type &>;
  // must have a size() returning something integral
  { v.size() } -> std::integral;
};

} // namespace athelas
