#include <cfenv>
#include <csignal>
#include <expected>
#include <filesystem>
#include <print>
#include <string>
#include <utility>
#include <vector>

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
  // Each entry is (dotted_key, lua_expr) collected from --key=value args.
  std::vector<std::pair<std::string, std::string>> overrides;
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
auto parse_input_options(std::span<char *> args)
    -> std::expected<CommandLineOptions, std::string> {
  CommandLineOptions opts;
  bool has_input = false;

  // handle parsing of -h / --help separately. Bare invocation also prints
  // help, but exits FAILURE (it's a usage error to omit -i).
  // Not really important but nice.
  const bool no_args = args.size() == 1;
  const bool get_help =
      args.size() >= 2 &&
      (std::strcmp(args[1], "-h") == 0 || std::strcmp(args[1], "--help") == 0);
  if (no_args || get_help) {
    std::println("# Usage: ./athelas [-h] [-i /path/to/input.lua] [-o "
                 "output_dir] [--key=value ...]");
    std::println("Options:");
    std::println("  -h, --help                Show this help message and exit");
    std::println("  -i <path>, --input=<path>");
    std::println("                            Path to the input .lua script "
                 "(Required)");
    std::println("  -o <dir>,  --output=<dir>");
    std::println("                            Directory where output files "
                 "will be saved (Default: ./)");
    std::println("  --<key>=<lua_expr>        Override a value in the input "
                 "deck. <key> is a dotted path");
    std::println("                            into the Lua config table; "
                 "<lua_expr> is parsed as Lua source.");
    std::println("                            Strings must be Lua-quoted: "
                 "--key='\"value\"'. Later values win.");

    std::println("Examples:");
    std::println("  ./athelas -i ../inputs/sod.lua");
    std::println("  ./athelas -i ../inputs/supernova.lua -o run/output");
    std::println("  ./athelas -i ../inputs/marshak.lua --problem.nx=256 "
                 "--radiation.newton.tol=1e-12");
    std::println("  ./athelas -i ../inputs/sedov.lua "
                 "--output.history.fn='\"sedov.hst\"'");
    std::println("  ./athelas -i ../inputs/marshak.lua "
                 "--bc.radiation.dirichlet_values_i='{{1.1e12, 0.0}}'");
    std::exit(get_help ? AthelasExitCodes::SUCCESS : AthelasExitCodes::FAILURE);
  }

  // Short flags (-i, -o) take the next argv as value. Long flags must use
  // --name=value; unrecognized --name=value pairs become input-deck overrides.
  for (std::size_t i = 1; i < args.size(); ++i) {
    std::string_view arg = args[i];

    if (arg == "-i") {
      if (i + 1 >= args.size()) {
        return std::unexpected("Missing input file after -i");
      }
      opts.input_file = args[++i];
      has_input = true;
      continue;
    }
    if (arg == "-o") {
      if (i + 1 >= args.size()) {
        return std::unexpected("Missing output directory after -o");
      }
      opts.output_dir = args[++i];
      continue;
    }

    // Handle long -- args
    if (!arg.starts_with("--")) {
      return std::unexpected(std::format("Unknown argument: {}", arg));
    }
    const auto eq = arg.find('=');
    if (eq == std::string_view::npos) {
      return std::unexpected(
          std::format("Long options require --name=value form: {}", arg));
    }
    const std::string_view name = arg.substr(0, eq);
    std::string value(arg.substr(eq + 1));

    if (name == "--input") {
      opts.input_file = std::move(value);
      has_input = true;
    } else if (name == "--output") {
      opts.output_dir = std::move(value);
    } else {
      // overrides
      std::string key(name.substr(2));
      if (key.empty()) {
        return std::unexpected(std::format("Override missing key: '{}'", arg));
      }
      opts.overrides.emplace_back(std::move(key), std::move(value));
    }
  }

  if (!has_input) {
    return std::unexpected("No input file passed! Use -i <path>");
  }

  return opts;
}
} // namespace

auto main(int argc, char **argv) -> int {

  auto input_result =
      parse_input_options({argv, static_cast<std::size_t>(argc)});
  if (!input_result) {
    std::println(std::cerr, "Error: {}", input_result.error());
    return AthelasExitCodes::FAILURE;
  }
  const auto &opts = input_result.value();

  namespace fs = std::filesystem;
  if (!fs::exists(opts.input_file)) {
    std::println(std::cerr, "Input file does not exist!");
    return athelas::AthelasExitCodes::FAILURE;
  }
  if (opts.output_dir != "./") {
    if (!fs::exists(opts.output_dir) || !fs::is_directory(opts.output_dir)) {
      std::println(std::cerr, "Output directory does not exist!");
      return athelas::AthelasExitCodes::FAILURE;
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
    const auto pin = std::make_shared<ProblemIn>(
        opts.input_file, opts.output_dir, opts.overrides);

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
