#include <cfenv>
#include <csignal>
#include <expected>
#include <print>
#include <string>

#include "Kokkos_Core.hpp"

#include "driver.hpp"
#include "main.hpp"
#include "problem_in.hpp"
#include "utils/error.hpp"

using athelas::Driver, athelas::AthelasExitCodes, athelas::ProblemIn,
    athelas::segfault_handler;

namespace {
auto parse_input_file(std::span<char *> args)
    -> std::expected<std::string, std::string> {
  for (std::size_t i = 1; i < args.size(); ++i) {
    std::string_view arg = args[i];
    if (arg == "-i" || arg == "--input") {
      if (i + 1 >= args.size()) {
        return std::unexpected("Missing input file after -i");
      }
      return std::string(args[i + 1]);
    }
  }
  return std::unexpected("No input file passed! Use -i <path>");
}
} // namespace

auto main(int argc, char **argv) -> int {
  auto input_result = parse_input_file({argv, static_cast<std::size_t>(argc)});
  if (!input_result) {
    std::println("Error: {}", input_result.error());
    return AthelasExitCodes::FAILURE;
  }

  std::string input_path = *input_result;

#ifdef ATHELAS_DEBUG
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  [[maybe_unused]] auto sig1 = signal(SIGSEGV, segfault_handler);
  [[maybe_unused]] auto sig2 = signal(SIGABRT, segfault_handler);
  [[maybe_unused]] auto sig3 = signal(SIGFPE, segfault_handler);
#endif

  std::println("# ----------------------------------------------------------");
  std::println("# Athelas running!");
  std::println(
      "# ----------------------------------------------------------\n");

  // create span of args
  // auto args = std::span( argv, static_cast<size_t>( argc ) );

  Kokkos::initialize(argc, argv);
  {
    // pin
    const auto pin = std::make_shared<ProblemIn>(input_path);

    // --- Create Driver ---
    Driver driver(pin);

    // --- Timer ---
    Kokkos::Timer timer_total;

    // --- execute driver ---
    Kokkos::Profiling::pushRegion("Driver::execute");
    driver.execute();
    Kokkos::Profiling::popRegion();

    // --- Finalize timer ---
    double const time = timer_total.seconds();
    std::println("# Athelas run complete! Elapsed time: {} seconds.", time);
  }
  Kokkos::finalize();

  return AthelasExitCodes::SUCCESS;
} // main
