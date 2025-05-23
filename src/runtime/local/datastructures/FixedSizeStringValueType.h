/*
 * Copyright 2024 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstddef>
#include <cstring>

/**
 * @brief A string value type with a maximum length of 15 characters.
 *
 * Each instance is backed by a 16-character buffer, whereby at least the last character must always be a null
 * character. The null-termination is required for some operations to work correctly (e.g., casting to a number).
 */
struct FixedStr16 {
    static const std::size_t N = 16;
    char buffer[N];

    // Default constructor
    FixedStr16() { std::fill(buffer, buffer + N, '\0'); }

    // Constructor from a C-style string
    FixedStr16(const char *str) {
        size_t len = std::strlen(str);
        if (len >= N) {
            throw std::length_error("string exceeds fixed buffer size");
        }
        std::copy(str, str + len, buffer);
        std::fill(buffer + len, buffer + N, '\0');
    }

    // Copy constructor
    FixedStr16(const FixedStr16 &other) { std::copy(other.buffer, other.buffer + N, buffer); }

    // Constructor from a std::string
    FixedStr16(const std::string &other) {
        size_t len = other.size();
        if (len >= N) {
            throw std::length_error("string exceeds fixed buffer size");
        }
        std::copy(other.begin(), other.end(), buffer);
        std::fill(buffer + len, buffer + N, '\0');
    }

    // Assignment operator
    FixedStr16 &operator=(const FixedStr16 &other) {
        if (this != &other) {
            std::copy(other.buffer, other.buffer + N, buffer);
        }
        return *this;
    }

    // Overriding the equality operator
    bool operator==(const FixedStr16 &other) const { return std::equal(buffer, buffer + N, other.buffer); }

    bool operator==(const char *str) const { return std::strncmp(buffer, str, sizeof(buffer)) == 0; }

    // Overriding the inequality operator
    bool operator!=(const FixedStr16 &other) const { return !(std::equal(buffer, buffer + N, other.buffer)); }

    bool operator!=(const char *str) const { return !(std::strncmp(buffer, str, sizeof(buffer)) == 0); }

    // Overriding the Less than operator
    bool operator<(const FixedStr16 &other) const { return std::strncmp(buffer, other.buffer, N) < 0; }

    // Overriding the Greater than operator
    bool operator>(const FixedStr16 &other) const { return std::strncmp(buffer, other.buffer, N) > 0; }

    // Concatenation operator
    friend std::string operator+(const FixedStr16 &lhs, const FixedStr16 &rhs) {
        std::string result(lhs.buffer);
        result.append(rhs.buffer);
        return result;
    }

    // Serialization function
    void serialize(std::vector<char> &outBuffer) const { outBuffer.insert(outBuffer.end(), buffer, buffer + N); }

    // Overload the output stream operator
    friend std::ostream &operator<<(std::ostream &os, const FixedStr16 &fs) {
        os.write(fs.buffer, N);
        return os;
    }

    // Size method
    size_t size() const { return std::strlen(buffer); }

    // Method to set the string
    void set(const char *str) {
        size_t len = std::strlen(str);
        if (len >= N) {
            throw std::length_error("string exceeds fixed buffer size");
        }
        std::transform(str, str + len, buffer, [](char c) { return c; });
        std::fill(buffer + len, buffer + N, '\0');
    }

    // C-string method for compatibility
    std::string to_string() const { return std::string(buffer, size()); }

    // Compare method similar to std::string::compare
    int compare(const FixedStr16 &other) const { return std::strncmp(buffer, other.buffer, N); }

    // Convert to lowercase
    FixedStr16 lower() const {
        FixedStr16 result;
        std::transform(buffer, buffer + N, result.buffer, [](unsigned char c) { return std::tolower(c); });
        return result;
    }

    // Convert to uppercase
    FixedStr16 upper() const {
        FixedStr16 result;
        std::transform(buffer, buffer + N, result.buffer, [](unsigned char c) { return std::toupper(c); });
        return result;
    }
};

// Specialize std::hash for FixedStr16 this is nessary to use FixedStr16 as a key in std::unordered_map
namespace std {
template <> struct hash<FixedStr16> {
    std::size_t operator()(const FixedStr16 &key) const {
        // Compute the hash of the fixed-size buffer
        return std::hash<std::string>()(std::string(key.buffer, key.N));
    }
};
} // namespace std
