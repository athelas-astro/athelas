/**
 * @file error.hpp
 * --------------
 *
 * @brief Error handling
 */

#pragma once

#include "Kokkos_Core.hpp"

#include <exception>
#include <execinfo.h>
#include <iostream>
#include <print>
#include <sstream>
#include <stacktrace>
#include <string>
#include <unistd.h>
#include <utility>

#include "utils/constants.hpp"

namespace athelas {

enum AthelasExitCodes {
  SUCCESS = 0,
  FAILURE = 1,
  PHYSICAL_CONSTRAINT_VIOLATION = 2,
  MEMORY_ERROR = 3,
  UNKNOWN_ERROR = 255
};

inline void print_backtrace() {
  std::cerr << std::stacktrace::current() << "\n";
}

[[noreturn]] inline void segfault_handler(int sig) {
  std::println(stderr, "Received signal {}", sig);
  print_backtrace();
  std::quick_exit(AthelasExitCodes::FAILURE);
}

class AthelasError : public std::exception {
 private:
  std::string m_message_;
  std::string m_function_;
  std::string m_file_;
  int m_line_;

 public:
  // Constructor with detailed error information
  explicit AthelasError(std::string message, const std::string &function = "",
                        const std::string &file = "", int line = 0)
      : m_message_(std::move(message)), m_function_(function), m_file_(file),
        m_line_(line) {}

  // Override what() to provide error details
  [[nodiscard]] auto what() const noexcept -> const char * override {
    static thread_local std::string full_message;
    std::ostringstream oss;

    oss << "!!! Athelas Error: " << m_message_ << "\n";

    if (!m_function_.empty()) {
      oss << "In function: " << m_function_ << "\n";
    }

    if (!m_file_.empty() && m_line_ > 0) {
      oss << "Location: " << m_file_ << ":" << m_line_ << "\n";
    }

    full_message = oss.str();
    return full_message.c_str();
  }
};

template <typename... Args>
[[noreturn]] inline void THROW_ATHELAS_ERROR(
    const std::string &message, const char *function = __builtin_FUNCTION(),
    const char *file = __builtin_FILE(), int line = __builtin_LINE()) {
  throw AthelasError(message, function, file, line);
}

inline void WARNING_ATHELAS(const std::string &message) {
  std::println("!!! Athelas Warning: {}", message);
  std::println(std::cerr, "!!! Athelas Warning: {}", message);
}

template <typename T>
void check_state(T state, const int ihi, const bool do_rad) {
  auto uCF = state.get_field("u_cf");
  const double c = constants::c_cgs;

  // Create host mirrors of the views
  auto uCF_h = Kokkos::create_mirror_view(uCF);

  // Copy data to host
  Kokkos::deep_copy(uCF_h, uCF);

  // Check state on host
  for (int ix = 1; ix <= ihi; ix++) {

    const double tau = uCF_h(ix, 0, 0); // cell averages checked
    const double vel = uCF_h(ix, 0, 1);
    const double e_m = uCF_h(ix, 0, 2);

    if (tau <= 0.0) {
      std::println("Error on cell {}", ix);
      THROW_ATHELAS_ERROR("Negative or zero density!");
    }
    if (std::isnan(tau)) {
      std::println("Error on cell {}", ix);
      THROW_ATHELAS_ERROR("Specific volume NaN!");
    }

    if (std::fabs(vel) >= c) {
      std::println("Error on cell {}", ix);
      THROW_ATHELAS_ERROR("Velocity reached or exceeded speed of light!");
    }
    if (std::isnan(vel)) {
      std::println("Error on cell {}", ix);
      THROW_ATHELAS_ERROR("Velocity NaN!");
    }

    if (e_m <= 0.0) {
      std::println("Error on cell {}", ix);
      THROW_ATHELAS_ERROR("Negative or zero specific total energy!");
    }
    if (std::isnan(e_m)) {
      std::println("Error on cell {}", ix);
      THROW_ATHELAS_ERROR("Specific energy NaN!");
    }

    if (do_rad) {
      const double e_rad = uCF_h(ix, 0, 3);
      const double f_rad = uCF_h(ix, 0, 4);

      if (std::isnan(e_rad)) {
        std::println("Error on cell {}", ix);
        THROW_ATHELAS_ERROR("Radiation energy NaN!");
      }
      if (e_rad <= 0.0) {
        std::println("Error on cell {}", ix);
        THROW_ATHELAS_ERROR("Negative or zero radiation energy density!");
      }

      if (std::isnan(f_rad)) {
        std::println("Error on cell {}", ix);
        THROW_ATHELAS_ERROR("Radiation flux NaN!");
      }
    }
  }
}

} // namespace athelas
