#pragma once

#include <algorithm>
#include <expected>
#include <fstream>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace athelas::io {

/**
 * @class Parser
 * @brief A modern C++23 CSV parser with robust parsing capabilities and
 * flexible data extraction
 *
 * The Parser class provides comprehensive CSV parsing functionality with
 * support for various delimiters, quoted fields, escape sequences, and
 * Python-friendly comment headers.
 *
 * @details
 * Key Features:
 * - **Generic parsing**: Handles any number of columns and various delimiters
 * - **Header support**: Automatically separates headers from data rows
 * - **Robust field parsing**: Correctly handles quoted fields, escaped quotes,
 * and embedded delimiters
 * - **Python-friendly**: Header must begin with '#'.
 * - **Type-safe extraction**: Template-based column extraction with automatic
 * type conversion
 * - **Modern C++23**: Uses std::expected, std::ranges, std::views for clean,
 * functional code
 * - **Memory efficient**: Provides lazy-evaluated views for large datasets
 * - **Error handling**: Comprehensive error reporting without exceptions
 *
 * @example Basic Parser usage
 * @code{.cpp}
 * // Parse from file
 * auto result = Parser::parse_file("data.csv");
 * if (result) {
 *     for (const auto& row : result->rows) {
 *         std::cout << row[0] << ", " << row[1] << std::endl;
 *     }
 * }
 *
 * // Parse from string with custom delimiter
 * std::string csv_data = "Name Age City\nJohn 30 NYC\nJane 25 LA";
 * auto space_result = Parser::parse_string(csv_data, ' ');
 *
 * // Extract typed columns
 * auto names = extract_column_by_name<std::string>(*space_result, "Name");
 * auto ages = extract_column_by_name<int>(*space_result, "Age");
 * @endcode
 *
 * @example Advanced Parser usage with Ranges
 * @code{.cpp}
 * auto result = Parser::parse_file("luminosities.csv");
 * if (result) {
 *     // Lazy evaluation - only processes when iterated
 *     auto lum_view = extract_column_view_by_index<double>(*result, 3);
 *     auto bright = lum_view
 *                 | std::views::filter([](double lum) { return lum > 1.0e42; })
 *                 | std::views::take(10);
 *
 *     for (const auto& lum : bright) {
 *         std::cout << lum << std::endl;
 *     }
 * }
 * @endcode
 */
class Parser {
 public:
  /**
   * @struct Row
   * @brief Holds a single row of parsed data.
   */
  struct Row {
    std::vector<std::string> columns;

    /**
     * @brief Access column by index
     * @param index 0-index column
     * @return const std::string& reference to column value or empty
     */
    auto operator[](size_t index) const -> const std::string & {
      return index < columns.size() ? columns[index] : empty_string;
    }

    [[nodiscard]] auto size() const noexcept -> size_t {
      return columns.size();
    }
    [[nodiscard]] auto empty() const noexcept -> bool {
      return columns.empty();
    }

    [[nodiscard]] auto begin() const noexcept { return columns.begin(); }
    [[nodiscard]] auto end() const { return columns.end(); }

   private:
    static inline const std::string empty_string{};
  };

  /**
   * @struct ParseResult
   * @brief Containr for parsed data with headers and rows
   */
  struct ParseResult {
    std::vector<std::string> headers;
    std::vector<Row> rows;
  };

  /**
   * @enum ParseError
   * @brief Error codes for parser failures.
   */
  enum class ParseError { FileNotFound, InvalidFormat, EmptyFile };

  // Parse from file
  static auto parse_file(const std::string &filename, char delimiter = ',',
                         char quote_char = '"')
      -> std::expected<ParseResult, ParseError> {
    std::ifstream file(filename);
    if (!file.is_open()) {
      return std::unexpected(ParseError::FileNotFound);
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    if (content.empty()) {
      return std::unexpected(ParseError::EmptyFile);
    }

    return parse_string(content, delimiter, quote_char);
  }

  // Parse from string
  static auto parse_string(const std::string &csv_content, char delimiter = ',',
                           char quote_char = '"',
                           bool strip_hash_from_header = true)
      -> std::expected<ParseResult, ParseError> {
    if (csv_content.empty()) {
      return std::unexpected(ParseError::EmptyFile);
    }

    ParseResult result;
    auto lines = csv_content | std::views::split('\n') |
                 std::views::transform([](auto &&range) {
                   return std::string_view{range.begin(), range.end()};
                 }) |
                 std::views::filter([](std::string_view line) {
                   return !line.empty() && line != "\r";
                 });

    bool first_line = true;
    for (auto line : lines) {
      // Remove carriage return if present
      if (!line.empty() && line.back() == '\r') {
        line.remove_suffix(1);
      }

      auto parsed_row = parse_line(line, delimiter, quote_char);

      // Handle header line. Extra logic to remove a # if it is
      // the first character.
      if (first_line) {
        first_line = false;
        auto headers = std::move(parsed_row);
        if (headers[0][0] != '#') {
          result.rows.emplace_back(Row{headers});
          continue;
        }

        // strip # from header if possible and desired
        if (strip_hash_from_header && !headers.empty() && !headers[0].empty() &&
            headers[0][0] == '#') {
          headers[0] = headers[0].substr(1);

          // trim any whitespace that may have been left
          headers[0] = trim(headers[0]);
        }

        result.headers = std::move(headers);
      } else {
        result.rows.emplace_back(Row{std::move(parsed_row)});
      }
    }

    return result;
  }

 private:
  static auto parse_line(std::string_view line, char delimiter, char quote_char)
      -> std::vector<std::string> {
    // Remove leading/trailing whitespace from line
    line = trim_view(line);

    std::vector<std::string> fields;
    std::string current_field;
    bool in_quotes = false;
    bool quote_encountered = false;

    for (size_t i = 0; i < line.size(); ++i) {
      char ch = line[i];

      if (ch == quote_char) {
        if (!in_quotes) {
          in_quotes = true;
          quote_encountered = true;
        } else if (i + 1 < line.size() && line[i + 1] == quote_char) {
          // Escaped quote
          current_field += quote_char;
          ++i; // Skip next quote
        } else {
          in_quotes = false;
        }
      } else if (ch == delimiter && !in_quotes) {
        // Trim whitespace only if field wasn't quoted
        if (!quote_encountered) {
          current_field = trim(current_field);
        }
        fields.push_back(std::move(current_field));
        current_field.clear();
        quote_encountered = false;
      } else {
        current_field += ch;
      }
    }

    // Add the last field
    if (!quote_encountered) {
      current_field = trim(current_field);
    }
    fields.push_back(std::move(current_field));

    return fields;
  }

  static auto trim(const std::string &str) -> std::string {
    auto start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
      return "";
    }

    auto end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
  }

  // trim leading and trailing whitespace
  static auto trim_view(std::string_view str) -> std::string_view {
    auto start = str.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
      return "";
    }

    auto end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
  }
}; // Parser

// --- Utility functions for extracting columns into containers ---
// TODO(astrobarker): Could be nice to make a "loadtxt" style wrapper
// that loads and gets.

template <typename T = std::string>
auto get_column_by_index(const Parser::ParseResult &data, size_t column_index)
    -> std::vector<T> {
  std::vector<T> column;
  column.reserve(data.rows.size());

  for (const auto &row : data.rows) {
    if (column_index < row.size()) {
      if constexpr (std::is_same_v<T, std::string>) {
        column.push_back(row[column_index]);
      } else if constexpr (std::is_arithmetic_v<T>) {
        try {
          if constexpr (std::is_integral_v<T>) {
            column.push_back(static_cast<T>(std::stoll(row[column_index])));
          } else {
            column.push_back(static_cast<T>(std::stod(row[column_index])));
          }
        } catch (const std::exception &) {
          column.push_back(T{}); // Default value on parse error
        }
      }
    } else {
      column.push_back(T{}); // Default value for missing columns
    }
  }
  return column;
}

// Gets a column by name.
// Find hheader index with given name, calls extract_column_by_index
template <typename T = std::string>
auto get_column_by_name(const Parser::ParseResult &data,
                        const std::string &column_name) -> std::vector<T> {
  auto it = std::ranges::find(data.headers, column_name);
  if (it == data.headers.end()) {
    return {}; // Column not found
  }

  size_t column_index = std::distance(data.headers.begin(), it);
  return get_column_by_index<T>(data, column_index);
}

template <typename T = std::string>
auto get_column_view_by_index(const Parser::ParseResult &data,
                              size_t column_index) {
  return data.rows |
         std::views::transform([column_index](const auto &row) -> T {
           if (column_index < row.size()) {
             if constexpr (std::is_same_v<T, std::string>) {
               return row[column_index];
             } else if constexpr (std::is_arithmetic_v<T>) {
               try {
                 if constexpr (std::is_integral_v<T>) {
                   return static_cast<T>(std::stoll(row[column_index]));
                 } else {
                   return static_cast<T>(std::stod(row[column_index]));
                 }
               } catch (const std::exception &) {
                 return T{};
               }
             }
           }
           return T{};
         });
}

// Extract multiple columns at once into a tuple of vectors
// Takes a ParseResult, std::array<size_t, N>{0, 1, ...N} (etc)
template <typename... Types>
auto get_columns_by_indices(const Parser::ParseResult &data,
                            std::array<size_t, sizeof...(Types)> indices) {
  return [&data, &indices]<size_t... I>(std::index_sequence<I...>) {
    return std::make_tuple(get_column_by_index<Types>(data, indices[I])...);
  }(std::make_index_sequence<sizeof...(Types)>{});
}

// Extract multiple columns at once into a tuple of vectors
// Instead of taking a std::array, as above, this takes variadic args
// Just pass in column indices!
template <typename... Types>
auto get_columns_by_indices(const Parser::ParseResult &data, size_t first_index,
                            auto... other_indices)
  requires(sizeof...(Types) == sizeof...(other_indices) + 1)
{
  std::array<size_t, sizeof...(Types)> indices{
      first_index, static_cast<size_t>(other_indices)...};
  return get_columns_by_indices<Types...>(data, indices);
}

} // namespace athelas::io
