#include "twofluid/json_input.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <cstdlib>

namespace twofluid {

// ---------------------------------------------------------------------------
// Minimal flat-JSON parser
// Supports: { "key": value, ... }
// value ::= number | string | true | false | null | array_of_numbers_or_strings
// No nested objects.
// ---------------------------------------------------------------------------

namespace {

static void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
}

static std::string parse_string(const std::string& s, size_t& i) {
    // i points at opening '"'
    ++i; // skip '"'
    std::string result;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
            ++i;
            switch (s[i]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case '/':  result += '/';  break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += s[i]; break;
            }
        } else {
            result += s[i];
        }
        ++i;
    }
    if (i < s.size()) ++i; // skip closing '"'
    return result;
}

static JsonValue parse_value(const std::string& s, size_t& i);

static JsonValue parse_array(const std::string& s, size_t& i) {
    // i points at '['
    ++i; // skip '['
    std::vector<double> nums;
    std::vector<std::string> strs;
    bool is_string_array = false;
    bool first = true;

    while (i < s.size()) {
        skip_ws(s, i);
        if (i >= s.size()) break;
        if (s[i] == ']') { ++i; break; }
        if (!first) {
            if (s[i] == ',') { ++i; skip_ws(s, i); }
        }
        first = false;
        if (i >= s.size()) break;

        if (s[i] == '"') {
            is_string_array = true;
            strs.push_back(parse_string(s, i));
        } else {
            // parse number
            size_t start = i;
            if (s[i] == '-') ++i;
            while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i]))
                                     || s[i] == '.' || s[i] == 'e'
                                     || s[i] == 'E' || s[i] == '+'
                                     || s[i] == '-')) ++i;
            double v = std::stod(s.substr(start, i - start));
            nums.push_back(v);
        }
    }

    if (is_string_array) return strs;
    return nums;
}

static JsonValue parse_value(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size()) throw std::runtime_error("Unexpected end of JSON");

    if (s[i] == '"') {
        return parse_string(s, i);
    }
    if (s[i] == '[') {
        return parse_array(s, i);
    }
    if (s.compare(i, 4, "true") == 0) {
        i += 4;
        return true;
    }
    if (s.compare(i, 5, "false") == 0) {
        i += 5;
        return false;
    }
    if (s.compare(i, 4, "null") == 0) {
        i += 4;
        return std::string("null");
    }
    // number
    size_t start = i;
    if (s[i] == '-') ++i;
    bool is_int = true;
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i]))
                              || s[i] == '.' || s[i] == 'e'
                              || s[i] == 'E' || s[i] == '+'
                              || (s[i] == '-' && i > start))) {
        if (s[i] == '.' || s[i] == 'e' || s[i] == 'E') is_int = false;
        ++i;
    }
    std::string tok = s.substr(start, i - start);
    if (is_int) {
        return static_cast<int>(std::stol(tok));
    }
    return std::stod(tok);
}

} // anonymous namespace

std::unordered_map<std::string, JsonValue> read_json_input(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open JSON file: " + filename);

    std::ostringstream oss;
    oss << ifs.rdbuf();
    std::string s = oss.str();

    std::unordered_map<std::string, JsonValue> result;
    size_t i = 0;
    skip_ws(s, i);

    if (i >= s.size() || s[i] != '{')
        throw std::runtime_error("JSON must start with '{'");
    ++i; // skip '{'

    while (i < s.size()) {
        skip_ws(s, i);
        if (i >= s.size()) break;
        if (s[i] == '}') { ++i; break; }
        if (s[i] == ',') { ++i; continue; }

        // key
        if (s[i] != '"')
            throw std::runtime_error("Expected string key at position " + std::to_string(i));
        std::string key = parse_string(s, i);

        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
            throw std::runtime_error("Expected ':' after key '" + key + "'");
        ++i; // skip ':'

        JsonValue val = parse_value(s, i);
        result[key] = val;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Typed accessors with defaults
// ---------------------------------------------------------------------------

double get_double(const std::unordered_map<std::string, JsonValue>& config,
                   const std::string& key, double default_val) {
    auto it = config.find(key);
    if (it == config.end()) return default_val;
    if (std::holds_alternative<double>(it->second)) return std::get<double>(it->second);
    if (std::holds_alternative<int>(it->second))    return static_cast<double>(std::get<int>(it->second));
    return default_val;
}

int get_int(const std::unordered_map<std::string, JsonValue>& config,
             const std::string& key, int default_val) {
    auto it = config.find(key);
    if (it == config.end()) return default_val;
    if (std::holds_alternative<int>(it->second))    return std::get<int>(it->second);
    if (std::holds_alternative<double>(it->second)) return static_cast<int>(std::get<double>(it->second));
    return default_val;
}

std::string get_string(const std::unordered_map<std::string, JsonValue>& config,
                        const std::string& key, const std::string& default_val) {
    auto it = config.find(key);
    if (it == config.end()) return default_val;
    if (std::holds_alternative<std::string>(it->second)) return std::get<std::string>(it->second);
    return default_val;
}

bool get_bool(const std::unordered_map<std::string, JsonValue>& config,
               const std::string& key, bool default_val) {
    auto it = config.find(key);
    if (it == config.end()) return default_val;
    if (std::holds_alternative<bool>(it->second)) return std::get<bool>(it->second);
    return default_val;
}

} // namespace twofluid
