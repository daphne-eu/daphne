#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define SHORT_STR_LEN 12
#define PREFIX_LEN 4

/**
 * @brief A string value type with a fixed memory size of 16 bytes, where the first 4 bytes
 * are reserved for the length of the string. When the length does not exceed 12 characters,
 * the string is stored in the remainig 12 bytes. Otherwise, 4 of the remaining 12 bytes
 * store a prefix of the string and the other 8 bytes a pointer to the string. 
 * This implementation uses a Union to realise the overall datastructure.
 */
struct Umbra_t {
    union {
        char short_str[SHORT_STR_LEN]; // For small strings
        struct {
            char prefix[PREFIX_LEN]; // Prefix for long strings
            char *ptr;               // Pointer to the long string
        } long_str;
    };
    uint32_t length; // Length of the string

    // Default constructor
    Umbra_t() : length(0) { std::fill(short_str, short_str + SHORT_STR_LEN, '\0'); }

    // Constructor from a C-style string
    Umbra_t(const char *str) {
        size_t len = std::strlen(str);
        if (len > UINT32_MAX) {
            throw std::length_error("String length exceeds maximum allowed");
        }
        length = static_cast<uint32_t>(len);
        if (length <= SHORT_STR_LEN) {
            std::memcpy(short_str, str, length);
            std::fill(short_str + length, short_str + SHORT_STR_LEN, '\0');
        } else {
            std::memcpy(long_str.prefix, str, PREFIX_LEN);
            long_str.ptr = new char[length + 1];
            std::memcpy(long_str.ptr, str, length);
            long_str.ptr[length] = '\0';
        }
    }

    // Constructor from a std::string
    Umbra_t(const std::string &str) {
        size_t len = str.size();
        if (len > UINT32_MAX) {
            throw std::length_error("String length exceeds maximum allowed");
        }
        length = static_cast<uint32_t>(len);
        if (length <= SHORT_STR_LEN) {
            str.copy(short_str, length);
            std::fill(short_str + length, short_str + SHORT_STR_LEN, '\0');
        } else {
            str.copy(long_str.prefix, PREFIX_LEN);
            long_str.ptr = new char[length + 1];
            str.copy(long_str.ptr, length);
            long_str.ptr[length] = '\0';
        }
    }

    // Copy constructor
    Umbra_t(const Umbra_t &other) {
        length = other.length;
        if (length <= SHORT_STR_LEN) {
            std::memcpy(short_str, other.short_str, length);
            std::fill(short_str + length, short_str + SHORT_STR_LEN, '\0');
        } else {
            std::memcpy(long_str.prefix, other.long_str.prefix, PREFIX_LEN);
            long_str.ptr = new char[length + 1];
            std::memcpy(long_str.ptr, other.long_str.ptr, length);
            long_str.ptr[length] = '\0';
        }
    }

    // Destructor
    ~Umbra_t() {
        if (length > SHORT_STR_LEN) {
            delete[] long_str.ptr;
            long_str.ptr = nullptr;
        }
    }

    // Assignment operator
    Umbra_t &operator=(const Umbra_t &other) {
        if (this != &other) {
            if (length > SHORT_STR_LEN) {
                delete[] long_str.ptr;
            }
            length = other.length;
            if (length <= SHORT_STR_LEN) {
                std::memcpy(short_str, other.short_str, length);
                std::fill(short_str + length, short_str + SHORT_STR_LEN, '\0');
            } else {
                std::memcpy(long_str.prefix, other.long_str.prefix, PREFIX_LEN);
                long_str.ptr = new char[length + 1];
                std::memcpy(long_str.ptr, other.long_str.ptr, length);
                long_str.ptr[length] = '\0';
            }
        }
        return *this;
    }

    // Copy
    Umbra_t(Umbra_t &&other) noexcept : length(other.length) {
        if (other.length <= SHORT_STR_LEN) {
            std::memcpy(short_str, other.short_str, SHORT_STR_LEN);
        } else {
            std::memcpy(long_str.prefix, other.long_str.prefix, PREFIX_LEN);
            long_str.ptr = other.long_str.ptr;
            other.long_str.ptr = nullptr;
            other.length = 0;
        }
    }

    // Assignment
    Umbra_t &operator=(Umbra_t &&other) noexcept {
        if (this != &other) {
            if (length > SHORT_STR_LEN) {
                delete[] long_str.ptr;
            }
            length = other.length;
            if (other.length <= SHORT_STR_LEN) {
                std::memcpy(short_str, other.short_str, SHORT_STR_LEN);
            } else {
                std::memcpy(long_str.prefix, other.long_str.prefix, PREFIX_LEN);
                long_str.ptr = other.long_str.ptr;
                other.long_str.ptr = nullptr;
                other.length = 0;
            }
        }
        return *this;
    }

    // Equality comparison with other Umbra Strings
    bool operator==(const Umbra_t &other) const {
        if (length != other.length) {
            return false;
        }
        if (length <= SHORT_STR_LEN) {
            return std::equal(short_str, short_str + length, other.short_str);
        } else {
            if (!std::equal(long_str.prefix, long_str.prefix + PREFIX_LEN, other.long_str.prefix)) {
                return false;
            }
            return std::equal(long_str.ptr, long_str.ptr + length, other.long_str.ptr);
        }
    }

    // Equality comparison with other C-style strings
    bool operator==(const char *str) const {
        size_t str_len = std::strlen(str);
        if (length != str_len) return false;
        if (length <= SHORT_STR_LEN)
            return std::equal(short_str, short_str + length, str);
        else
            return std::equal(long_str.ptr, long_str.ptr + length, str);
    }

    // Inequality comparison with other Umbra Strings
    bool operator!=(const Umbra_t &other) const { return !(*this == other); }

    // Inequality comparison with other C-style strings
    bool operator!=(const char *str) const { return !(*this == str); }

    // Less-than comparison with other Umbra Strings
    bool operator<(const Umbra_t &other) const {
        uint32_t min_length = std::min(length, other.length);
        int cmp;
        if (length <= SHORT_STR_LEN && other.length <= SHORT_STR_LEN) {
            cmp = std::memcmp(short_str, other.short_str, min_length);
        } else if (length <= SHORT_STR_LEN) {
            cmp = std::memcmp(short_str, other.long_str.prefix, PREFIX_LEN);
            if (cmp == 0) {
                cmp = std::memcmp(short_str, other.long_str.ptr, min_length);
            }
        } else if (other.length <= SHORT_STR_LEN) {
            cmp = std::memcmp(long_str.prefix, other.short_str, PREFIX_LEN);
            if (cmp == 0) {
                cmp = std::memcmp(long_str.ptr, other.short_str, min_length);
            }
        } else {
            cmp = std::memcmp(long_str.prefix, other.long_str.prefix, PREFIX_LEN);
            if (cmp == 0) {
                cmp = std::memcmp(long_str.ptr, other.long_str.ptr, min_length);
            }
        }
        if (cmp == 0) {
            return length < other.length;
        }
        return cmp < 0;
    }

    // Less-than comparison with other C-style strings
    bool operator<(const char *str) const {
        uint32_t str_len = std::strlen(str);
        uint32_t min_length = std::min(length, str_len);
        int cmp;
        if (length <= SHORT_STR_LEN) {
            cmp = std::memcmp(short_str, str, min_length);
        } else {
            cmp = std::memcmp(long_str.prefix, str, PREFIX_LEN);
            if (cmp == 0) {
                cmp = std::memcmp(long_str.ptr, str, min_length);
            }
        }
        if (cmp == 0) {
            return length < str_len;
        }
        return cmp < 0;
    }

    // Greater-than comparison with other Umbra Strings
    bool operator>(const Umbra_t &other) const {
        uint32_t min_length = std::min(length, other.length);
        int cmp;
        if (length <= SHORT_STR_LEN && other.length <= SHORT_STR_LEN) {
            cmp = std::memcmp(short_str, other.short_str, min_length);
        } else if (length <= SHORT_STR_LEN) {
            cmp = std::memcmp(short_str, other.long_str.prefix, PREFIX_LEN);
            if (cmp == 0) {
                cmp = std::memcmp(short_str, other.long_str.ptr, min_length);
            }
        } else if (other.length <= SHORT_STR_LEN) {
            cmp = std::memcmp(long_str.prefix, other.short_str, PREFIX_LEN);
            if (cmp == 0) {
                cmp = std::memcmp(long_str.ptr, other.short_str, min_length);
            }
        } else {
            cmp = std::memcmp(long_str.prefix, other.long_str.prefix, PREFIX_LEN);
            if (cmp == 0) {
                cmp = std::memcmp(long_str.ptr, other.long_str.ptr, min_length);
            }
        }
        if (cmp == 0) {
            return length > other.length;
        }
        return cmp > 0;
    }

    // Greater-than comparison with other C-style strings
    bool operator>(const char *str) const {
        uint32_t str_len = std::strlen(str);
        uint32_t min_length = std::min(length, str_len);
        int cmp;
        if (length <= SHORT_STR_LEN) {
            cmp = std::memcmp(short_str, str, min_length);
        } else {
            cmp = std::memcmp(long_str.prefix, str, PREFIX_LEN);
            if (cmp == 0) {
                cmp = std::memcmp(long_str.ptr, str, min_length);
            }
        }
        if (cmp == 0) {
            return length > str_len;
        }
        return cmp > 0;
    }

    operator std::string() const { return this->to_string(); }

    // Concatenation Operation with other Umbra Strings.

    Umbra_t operator+(const Umbra_t &other) const {
        Umbra_t result;
        result.length = this->length + other.length;

        if (result.length <= SHORT_STR_LEN) {
            std::memcpy(result.short_str, this->short_str, this->length);
            std::memcpy(result.short_str + this->length, other.short_str, other.length);
        } else {
            char *new_str = new char[result.length + 1];

            if (this->length <= SHORT_STR_LEN) {
                std::memcpy(new_str, this->short_str, this->length);
            } else {
                std::memcpy(new_str, this->long_str.ptr, this->length);
            }

            if (other.length <= SHORT_STR_LEN) {
                std::memcpy(new_str + this->length, other.short_str, other.length);
            } else {
                std::memcpy(new_str + this->length, other.long_str.ptr, other.length);
            }

            new_str[result.length] = '\0';

            std::memcpy(result.long_str.prefix, new_str, PREFIX_LEN);
            result.long_str.ptr = new_str;
        }

        return result;
    }

    // Concatenation Operation with other C-style strings.
    Umbra_t operator+(const char *str) const {
        uint32_t str_length = std::strlen(str);
        uint32_t new_length = length + str_length;
        char *new_str = new char[new_length + 1];

        if (length <= SHORT_STR_LEN) {
            std::memcpy(new_str, short_str, length);
        } else {
            std::memcpy(new_str, long_str.ptr, length);
        }

        std::memcpy(new_str + length, str, str_length);

        new_str[new_length] = '\0';
        Umbra_t result(new_str);
        delete[] new_str;
        return result;
    }

    // Serialize method
    void serialize(std::vector<char> &outBuffer) const {
        outBuffer.reserve(4 + length);

        outBuffer.push_back(static_cast<char>((length >> 0) & 0xFF));
        outBuffer.push_back(static_cast<char>((length >> 8) & 0xFF));
        outBuffer.push_back(static_cast<char>((length >> 16) & 0xFF));
        outBuffer.push_back(static_cast<char>((length >> 24) & 0xFF));

        if (length <= SHORT_STR_LEN) {
            outBuffer.insert(outBuffer.end(), short_str, short_str + length);
        } else {
            outBuffer.insert(outBuffer.end(), long_str.ptr, long_str.ptr + length);
        }
    }

    // size method 
    inline size_t size() const { return length; }

    // Method to check if string is stored in long format
    inline bool is_long() const { return length > SHORT_STR_LEN; }

    // method to get string
    inline const char *get() const { return is_long() ? long_str.ptr : short_str; }

    // Method to set the String
    void set(const char *str) {
        size_t len = std::strlen(str);

        if (len > UINT32_MAX) {
            throw std::length_error("String length exceeds maximum allowed");
        }

        if (is_long()) {
            delete[] long_str.ptr; // Clean up old string if previously long
        }

        length = static_cast<uint32_t>(len);

        if (length <= SHORT_STR_LEN) {
            std::memcpy(short_str, str, length);
            std::fill(short_str + length, short_str + SHORT_STR_LEN, '\0');

        } else {
            std::memcpy(long_str.prefix, str, PREFIX_LEN);
            long_str.ptr = new char[length + 1];
            std::memcpy(long_str.ptr, str, length);
            long_str.ptr[length] = '\0';
        }
    }

    // Method to convert Umbra to std::string
    std::string to_string() const {
        if (length <= SHORT_STR_LEN) {
            return std::string(short_str, length);
        }
        return std::string(long_str.ptr, size());
    }

    // Output stream operator
    friend std::ostream &operator<<(std::ostream &os, const Umbra_t &str) {
        if (str.length <= SHORT_STR_LEN) {
            os.write(str.short_str, str.length);
        } else {
            os.write(str.long_str.ptr, str.length);
        }

        return os;
    }

    // Compare method similar to std::string::compare
    int compare(const char *str) const {
        uint32_t str_len = std::strlen(str);
        uint32_t min_length = std::min(length, str_len);
        int cmp;
        if (length <= SHORT_STR_LEN) {
            cmp = std::memcmp(short_str, str, min_length);
        } else {
            cmp = std::memcmp(long_str.prefix, str, PREFIX_LEN);
            if (cmp == 0) {
                cmp = std::memcmp(long_str.ptr, str, min_length);
            }
        }
        return cmp;
    }

    // Convert to lowercase
    Umbra_t lower() const {
        Umbra_t result(*this);
        if (length <= SHORT_STR_LEN) {
            for (uint32_t i = 0; i < length; i++) {
                result.short_str[i] = static_cast<char>(std::tolower(result.short_str[i]));
            }
        } else {
            char *str = reinterpret_cast<char *>(result.long_str.ptr);
            for (uint32_t i = 0; i < length; i++) {
                str[i] = static_cast<char>(std::tolower(str[i]));
            }
            for (int i = 0; i < PREFIX_LEN; i++) {
                result.long_str.prefix[i] = static_cast<char>(std::tolower(result.long_str.prefix[i]));
            }
        }
        return result;
    }

    // Convert to uppercase
    Umbra_t upper() const {
        Umbra_t result(*this);
        if (length <= SHORT_STR_LEN) {
            for (uint32_t i = 0; i < length; i++) {
                result.short_str[i] = static_cast<char>(std::toupper(result.short_str[i]));
            }
        } else {
            char *str = reinterpret_cast<char *>(result.long_str.ptr);
            for (uint32_t i = 0; i < length; i++) {
                str[i] = static_cast<char>(std::toupper(str[i]));
            }
            for (int i = 0; i < PREFIX_LEN; i++) {
                result.long_str.prefix[i] = static_cast<char>(std::toupper(result.long_str.prefix[i]));
            }
        }
        return result;
    }
} __attribute__((aligned(8)));


// Hash function for use in unordered map
namespace std {
template <> struct hash<Umbra_t> {
    size_t operator()(const Umbra_t &key) const {
        return std::hash<std::string_view>()(std::string_view(key.get(), key.size()));
    }
};
} // namespace std