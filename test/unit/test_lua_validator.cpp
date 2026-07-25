#include <string>

#include <catch2/catch_test_macros.hpp>

#include "pgen/lua_validator.hpp"
#include "utils/error.hpp"

namespace {

auto error_contains(const athelas::AthelasError &error,
                    const std::string &message) -> bool {
  return std::string(error.what()).contains(message);
}

} // namespace

TEST_CASE("Lua validator rejects unrecognized fields", "[lua_validator]") {
  sol::state lua;
  lua.open_libraries(sol::lib::base);

  sol::table schema = lua.script(R"(
    return {
      known = { doc = "A known field." },
      nested = {
        child = { doc = "A known nested field." },
      },
    }
  )");

  SECTION("top-level field") {
    sol::table config = lua.script("return { typo = true }");
    athelas::Validator validator(schema);

    try {
      validator.validate(config);
      FAIL("Expected validation to reject an unknown top-level field");
    } catch (const athelas::AthelasError &error) {
      REQUIRE(error_contains(error, "Unrecognized field: typo"));
    }
  }

  SECTION("nested field") {
    sol::table config = lua.script("return { nested = { typo = true } }");
    athelas::Validator validator(schema);

    try {
      validator.validate(config);
      FAIL("Expected validation to reject an unknown nested field");
    } catch (const athelas::AthelasError &error) {
      REQUIRE(error_contains(error, "Unrecognized field: nested.typo"));
    }
  }

  SECTION("non-string key") {
    sol::table config = lua.script("return { [1] = true }");
    athelas::Validator validator(schema);

    try {
      validator.validate(config);
      FAIL("Expected validation to reject a non-string key");
    } catch (const athelas::AthelasError &error) {
      REQUIRE(error_contains(error,
                             "Non-string key in configuration table: <root>"));
    }
  }
}

TEST_CASE("Lua validator permits explicitly open tables", "[lua_validator]") {
  sol::state lua;
  lua.open_libraries(sol::lib::base);

  sol::table schema = lua.script(R"(
    return {
      defaults_to_five = { default = 5, doc = "A defaulted field." },
      required_value = { required = true, doc = "A required field." },
      open = { allow_unknown = true },
    }
  )");

  SECTION("accepts arbitrary contents below allow_unknown") {
    sol::table config = lua.script(R"(
      return {
        required_value = true,
        open = { arbitrary = true, [1] = "also permitted" },
      }
    )");
    athelas::Validator validator(schema);

    validator.validate(config);

    const sol::object default_value = config["defaults_to_five"];
    REQUIRE(default_value.as<int>() == 5);
  }

  SECTION("retains required-field validation") {
    sol::table config = lua.script("return {} ");
    athelas::Validator validator(schema);

    try {
      validator.validate(config);
      FAIL("Expected validation to reject a missing required field");
    } catch (const athelas::AthelasError &error) {
      REQUIRE(error_contains(error, "Missing required field: required_value"));
    }
  }
}
