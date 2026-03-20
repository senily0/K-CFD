#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>

namespace twofluid {

/// Simple JSON value type: scalar number, string, bool, or array of numbers.
using JsonValue = std::variant<double, int, bool, std::string,
                                std::vector<double>, std::vector<std::string>>;

/// Read a flat JSON input file and return key-value pairs.
/// Supports: numbers, strings, booleans, arrays of numbers or strings.
/// Nested objects are not supported.
std::unordered_map<std::string, JsonValue> read_json_input(const std::string& filename);

/// Get a double value, returning default_val if key is missing or wrong type.
double get_double(const std::unordered_map<std::string, JsonValue>& config,
                   const std::string& key, double default_val);

/// Get an int value, returning default_val if key is missing or wrong type.
int get_int(const std::unordered_map<std::string, JsonValue>& config,
             const std::string& key, int default_val);

/// Get a string value, returning default_val if key is missing or wrong type.
std::string get_string(const std::unordered_map<std::string, JsonValue>& config,
                        const std::string& key, const std::string& default_val);

/// Get a bool value, returning default_val if key is missing or wrong type.
bool get_bool(const std::unordered_map<std::string, JsonValue>& config,
               const std::string& key, bool default_val);

} // namespace twofluid
