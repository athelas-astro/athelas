#include <cfenv>
#include <csignal>
#include <expected>
#include <filesystem>
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
struct CommandLineOptions {
  std::string input_file;
  std::string output_dir = "./";
};

/*
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
*/
auto parse_input_file(std::span<char *> args)
    -> std::expected<CommandLineOptions, std::string> {
  CommandLineOptions opts;
  bool has_input = false;

  for (std::size_t i = 1; i < args.size(); ++i) {
    std::string_view arg = args[i];

    if (arg == "-i" || arg == "--input") {
      if (i + 1 >= args.size()) {
        return std::unexpected("Missing input file after -i");
      }
      opts.input_file = args[i + 1];
      has_input = true;
      ++i; // Skip the next argument since we consumed it
    } else if (arg == "-o" || arg == "--output-dir") {
      if (i + 1 >= args.size()) {
        return std::unexpected("Missing output directory after -o");
      }
      opts.output_dir = args[i + 1];
      ++i; // Skip the next argument since we consumed it
    } else {
      return std::unexpected(std::format("Unknown argument: {}", arg));
    }
  }

  if (!has_input) {
    return std::unexpected("No input file passed! Use -i <path>");
  }

  return opts;
}
} // namespace

auto main(int argc, char **argv) -> int {
  auto input_result = parse_input_file({argv, static_cast<std::size_t>(argc)});
  if (!input_result) {
    std::println("Error: {}", input_result.error());
    return AthelasExitCodes::FAILURE;
  }
  const auto &opts = input_result.value();

  namespace fs = std::filesystem;
  if (!fs::exists(opts.input_file)) {
    athelas::throw_athelas_error("Input file does not exist!");
  }
  if (opts.output_dir != "./") {
    if (!fs::exists(opts.output_dir) || !fs::is_directory(opts.output_dir)) {
      athelas::throw_athelas_error("Output directory does not exist!");
    }
  }

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
    const auto pin =
        std::make_shared<ProblemIn>(opts.input_file, opts.output_dir);

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
