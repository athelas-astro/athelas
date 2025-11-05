#include <print>

#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"

TEST_CASE("Parser parses valid text file correctly", "[parser]") {

  SECTION("Parsing CSV data") {
    std::string sample_csv = R"(# Name,Age,City
  "John Doe",30,"New York"
  "Jane Smith",25,Los Angeles
  "Bob",35,"Underworld"
  "A",200,Boston)";

    auto result = Parser::parse_string(sample_csv, ',');

    SECTION("Comma separated text successfully parsed") {
      REQUIRE(result);
      REQUIRE(result->rows.size() == 4);
      REQUIRE(result->headers.size() == 3); // items in header
    }

    // Unnecessary but could use useful
    if (result) {
      std::cout << std::format("Successfully parsed {} rows with {} columns\n",
                               result->rows.size(), result->headers.size());

      // Demonstrate range-based access
      std::cout << "\n=== Range-based iteration ===\n";
      for (const auto &[idx, row] : std::views::enumerate(result->rows)) {
        std::cout << std::format("Row {}: ", idx + 1);
        for (const auto &field : row) {
          std::cout << std::format("[{}] ", field);
        }
        std::cout << "\n";
      }

      // Demonstrate filtering with C++23 ranges
      std::cout << "\n=== Filtered results (age > 28) ===\n";
      auto age_header_idx =
          std::ranges::find(result->headers, "Age") - result->headers.begin();

      auto filtered_rows =
          result->rows | std::views::filter([age_header_idx](const auto &row) {
            if (age_header_idx < row.size()) {
              try {
                return std::stoi(row[age_header_idx]) > 28;
              } catch (...) {
                return false;
              }
            }
            return false;
          });

      for (const auto &row : filtered_rows) {
        std::cout << std::format("Name: {}, Age: {}, City: {}\n", row[0],
                                 row[1], row[2]);
      }

    } else {
      std::cout << "Failed to parse CSV\n";
    }
  }

  SECTION("Parsing space delimited data") {
    std::string sample_csv = R"(# Name Age City
  "Aragorn" 87 "Rivendell"
  "Frodo Baggins" 50 Hobbiton
  "Meriadoc Brandybuck" 37 "The Shire"
  Sauron 3000 Mordor)";

    auto result = Parser::parse_string(sample_csv, ' ');

    SECTION("Space separated text successfully parsed") {
      REQUIRE(result);
      REQUIRE(result->rows.size() == 4);
      REQUIRE(result->headers.size() == 3); // items in header
    }

    if (result) {
      std::cout << std::format("Successfully parsed {} rows with {} columns\n",
                               result->rows.size(), result->headers.size());
      print_parser_data(*result);

      // Demonstrate range-based access
      std::cout << "\n=== Range-based iteration ===\n";
      for (const auto &[idx, row] : std::views::enumerate(result->rows)) {
        std::cout << std::format("Row {}: ", idx + 1);
        for (const auto &field : row) {
          std::cout << std::format("[{}] ", field);
        }
        std::cout << "\n";
      }

      // Demonstrate filtering with C++23 ranges
      std::cout << "\n=== Filtered results (age > 28) ===\n";
      auto age_header_idx =
          std::ranges::find(result->headers, "Age") - result->headers.begin();

      auto filtered_rows =
          result->rows | std::views::filter([age_header_idx](const auto &row) {
            if (age_header_idx < row.size()) {
              try {
                return std::stoi(row[age_header_idx]) > 28;
              } catch (...) {
                return false;
              }
            }
            return false;
          });

      for (const auto &row : filtered_rows) {
        std::cout << std::format("Name: {}, Age: {}, City: {}\n", row[0],
                                 row[1], row[2]);
      }

    } else {
      std::cout << "Failed to parse CSV\n";
    }
  }

  SECTION("Parse file") {
    // This is pretty awful but oh well.
    const std::string fn = "../../../test/unit/test.dat";

    auto result = Parser::parse_file(fn, ' ');
    REQUIRE(result);

    auto [col1, col2, col3] = get_columns_by_indices<int, int, int>(
        *result, std::array<size_t, 3>{0, 1, 2});

    SECTION("File parsed correctly") {
      REQUIRE(col1[0] == 1);
      REQUIRE(col1[1] == 3);
      REQUIRE(col2[0] == 2);
      REQUIRE(col2[1] == 4);
      REQUIRE(col3[0] == 3);
      REQUIRE(col3[1] == 5);
    }

    SECTION("Other parsing utilities work") {
      auto col1 = get_column_by_name<int>(*result, "header1");
      REQUIRE(col1[0] == 1);
      REQUIRE(col1[1] == 3);

      auto col1_view = get_column_view_by_index(*result, 0);
      REQUIRE(col1[0] == 1);
      REQUIRE(col1[1] == 3);
    }
  }
}
