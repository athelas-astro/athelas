#pragma once

#include <concepts>

namespace athelas {

// Define a concept that ensures subtraction is valid
template <typename T>
concept Subtractable = requires(T a, T b) {
  { a - b } -> std::convertible_to<T>;
};

// A concept: type must be multiplicative and copyable
template <class T>
concept Multipliable = requires(T a, T b) {
  { a * b } -> std::same_as<T>;
  { a = a * b };
};

} // namespace athelas
