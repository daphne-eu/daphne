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
 #include <cctype>
 #include <cstddef>
 #include <cstring>
 #include <functional>
 #include <iostream>
 #include <stdexcept>
 #include <string>
 #include <vector>
 
 /**
  * @brief A string value type with a maximum length of N = 2^n = {16, 32, 64, 128, 256} characters.
  *
  * Each instance is backed by a N-character buffer, whereby at least the last character must always be a null
  * character. The null-termination is required for some operations to work correctly (e.g., casting to a number).
  */
 template <std::size_t N> class FixedStr {
     static_assert(N > 0, "Capacity must be greater than 0.");
     static_assert((N & (N - 1)) == 0, "Capacity must be a power of 2.");
 
   public:
     char buffer[N];
 
     // Default Constructor
     FixedStr() { std::fill(buffer, buffer + N, '\0'); }
 
     // Constructor from a C-style string
     FixedStr(const char *str) {
         std::size_t len = std::strlen(str);
         if (len >= N)
             throw std::length_error("string exceeds fixed buffer size");
         std::copy(str, str + len, buffer);
         std::fill(buffer + len, buffer + N, '\0');
     }
 
     // Copy constructor
     FixedStr(const FixedStr &other) { std::copy(other.buffer, other.buffer + N, buffer); }
 
     // Constructor from a std::string
     FixedStr(const std::string &str) {
         std::size_t len = str.size();
         if (len >= N)
             throw std::length_error("string exceeds fixed buffer size");
         std::copy(str.begin(), str.end(), buffer);
         std::fill(buffer + len, buffer + N, '\0');
     }
 
     // Assignment operator
     FixedStr &operator=(const FixedStr &other) {
         if (this != &other) {
             std::copy(other.buffer, other.buffer + N, buffer);
         }
         return *this;
     }
 
     // Overriding the comparison operators
     bool operator==(const FixedStr &other) const { return std::equal(buffer, buffer + N, other.buffer); }
     bool operator==(const char *str) const { return std::strncmp(buffer, str, N) == 0; }
 
     bool operator!=(const FixedStr &other) const { return !(*this == other); }
     bool operator!=(const char *str) const { return !(*this == str); }
 
     bool operator<(const FixedStr &other) const { return std::strncmp(buffer, other.buffer, N) < 0; }
 
     bool operator>(const FixedStr &other) const { return std::strncmp(buffer, other.buffer, N) > 0; }
 

     // Concatenation operator
     friend std::string operator+(const FixedStr &lhs, const FixedStr &rhs) {
         std::string result(lhs.buffer);
         result.append(rhs.buffer);
         return result;
     }
 
     // Serialization function
     void serialize(std::vector<char> &outBuffer) const { outBuffer.insert(outBuffer.end(), buffer, buffer + N); }
 
     // Overload the output stream operator
     friend std::ostream &operator<<(std::ostream &os, const FixedStr &fs) {
         os << fs.to_string();
         return os;
     }
 
     // Size method
     std::size_t size() const { return std::strlen(buffer); }

     // Method to set the string
     void set(const char *str) {
         std::size_t len = std::strlen(str);
         if (len >= N)
             throw std::length_error("string exceeds fixed buffer size");
         std::copy(str, str + len, buffer);
         std::fill(buffer + len, buffer + N, '\0');
     }
 
     // C-string method for compatibility
     std::string to_string() const { return std::string(buffer, size()); }
 
     // Compare method similar to std::string::compare
     int compare(const FixedStr &other) const { return std::strncmp(buffer, other.buffer, N); }
 
     // Lowercase method
     FixedStr lower() const {
         FixedStr result(*this);
         std::transform(result.buffer, result.buffer + N, result.buffer,
                        [](unsigned char c) { return std::tolower(c); });
         return result;
     }
 
     // Uppercase method
     FixedStr upper() const {
         FixedStr result(*this);
         std::transform(result.buffer, result.buffer + N, result.buffer,
                        [](unsigned char c) { return std::toupper(c); });
         return result;
     }
 };
 
// define hash for use in unordered map
 namespace std {
 template <std::size_t N> struct hash<FixedStr<N>> {
     std::size_t operator()(const FixedStr<N> &key) const { return std::hash<std::string>()(key.to_string()); }
 };
 } // namespace std
 
 // Type aliases for common fixed string sizes.
 using FixedStr16 = FixedStr<16>;
 using FixedStr32 = FixedStr<32>;
 using FixedStr64 = FixedStr<64>;
 using FixedStr128 = FixedStr<128>;
 using FixedStr256 = FixedStr<256>;