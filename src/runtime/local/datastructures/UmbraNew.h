#pragma once // ifndef

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define BUFFER_LEN 12
#define PREFIX_LEN 4

/**
 * @brief A string value type with a fixed memory size of 16 bytes, where the first 4 bytes
 * are reserved for the length of the string. When the length does not exceed 12 characters,
 * the string is stored in the remainig 12 bytes. Otherwise, 4 of the remaining 12 bytes
 * store a prefix of the string and the other 8 bytes a pointer to the string. 
 * This implementation uses a uint8_t buffer to store the remaining 12 bytes in comparison to the union of the other Umbra implementation.
 */
class NewUmbra_t {
    static_assert(BUFFER_LEN >= PREFIX_LEN + sizeof(uintptr_t), "Buffer too small to store pointer.");

  private:
    uint32_t length; // Length of the string
    uint8_t buffer[BUFFER_LEN]; // buffer to hold string if <= 12 bytes or prefix and pointer

    bool is_long() const { return length > BUFFER_LEN; }

    void set_pointer(const std::string &full_str) {
        std::string *ptr = new std::string(full_str);

        uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
        std::memcpy(buffer + PREFIX_LEN, &address, sizeof(uintptr_t));
    }

    std::string *get_pointer() const {
        uintptr_t address;
        std::memcpy(&address, buffer + PREFIX_LEN, sizeof(uintptr_t));
        return reinterpret_cast<std::string *>(address);
    }

    void set_string(const char *str) {
        std::copy(str, str + PREFIX_LEN, buffer);
        set_pointer(std::string(str));
    }

    void clear_pointer() { std::memset(buffer + PREFIX_LEN, 0, sizeof(uintptr_t)); }

  public:
    // Default constructor
    NewUmbra_t() : length(0) { std::fill(buffer, buffer + BUFFER_LEN, static_cast<uint8_t>('\0')); }

    // Constructor from a C-style string
    NewUmbra_t(const char *str) {
        length = static_cast<uint32_t>(std::strlen(str));
        if (length > UINT32_MAX) {
            throw std::length_error("String exceeds fixed buffer size");
        }
        if (length <= BUFFER_LEN) {
            std::copy(str, str + length, buffer);
            std::fill(buffer + length, buffer + BUFFER_LEN, static_cast<uint8_t>('\0'));
        } else {
            set_string(str);
        }
    }

    // Copy constructor
    NewUmbra_t(const NewUmbra_t &other) : length(other.length) {
        if (length <= BUFFER_LEN) {
            std::copy(other.buffer, other.buffer + BUFFER_LEN, buffer);
        } else {
            std::copy(other.buffer, other.buffer + PREFIX_LEN, buffer);
            std::string *original_ptr = other.get_pointer();
            if (original_ptr) {
                std::string *new_ptr = new std::string(*original_ptr);
                uintptr_t new_address = reinterpret_cast<uintptr_t>(new_ptr);
                std::memcpy(buffer + PREFIX_LEN, &new_address, sizeof(uintptr_t));
            }
        }
    }

    // Constructor from a std::string
    NewUmbra_t(const std::string &str) {
        length = static_cast<uint32_t>(str.size());
        if (length > UINT32_MAX) {
            throw std::length_error("String exceeds fixed buffer size");
        }
        if (length <= BUFFER_LEN) {
            std::copy(str.begin(), str.begin() + length, buffer);
            std::fill(buffer + length, buffer + BUFFER_LEN, static_cast<uint8_t>('\0'));
        } else {
            set_string(str.c_str());
        }
    }

    // Copy constructor
    NewUmbra_t(NewUmbra_t &&other) noexcept : length(other.length) {
        if (!other.is_long()) {
            std::copy(other.buffer, other.buffer + BUFFER_LEN, buffer);
        } else {
            std::memcpy(buffer, other.buffer, sizeof(uint8_t) * BUFFER_LEN);
            other.length = 0;
            other.clear_pointer();
        }
    }

    // Assignment operator
    NewUmbra_t &operator=(const NewUmbra_t &other) {
        if (this != &other) {
            if (is_long()) {
                delete get_pointer();
                clear_pointer();
            }

            length = other.length;
            if (!other.is_long()) {
                std::copy(other.buffer, other.buffer + BUFFER_LEN, buffer);
            } else {

                std::copy(other.buffer, other.buffer + PREFIX_LEN, buffer);
                std::string *original_ptr = other.get_pointer();
                if (original_ptr) {
                    std::string *new_ptr = new std::string(*original_ptr);
                    uintptr_t new_address = reinterpret_cast<uintptr_t>(new_ptr);
                    std::memcpy(buffer + PREFIX_LEN, &new_address, sizeof(uintptr_t));
                }
            }
        }
        return *this;
    }

    NewUmbra_t &operator=(NewUmbra_t &&other) noexcept {
        if (this != &other) {
            if (is_long()) {
                delete get_pointer();
                clear_pointer();
            }

            length = other.length;
            std::copy(other.buffer, other.buffer + BUFFER_LEN, buffer);

            if (other.is_long()) {

                std::memcpy(buffer + PREFIX_LEN, other.buffer + PREFIX_LEN, sizeof(uintptr_t));
                other.length = 0;
                other.clear_pointer();
            }
        }
        return *this;
    }

    // Destructor
    ~NewUmbra_t() {
        if (is_long()) {
            delete get_pointer();
            clear_pointer();
        }
    }

    // Comparison operators
    bool operator==(const NewUmbra_t &other) const {
        if (length != other.length)
            return false;

        if (!is_long()) {
            return std::equal(buffer, buffer + BUFFER_LEN, other.buffer);
        } else {
            if (!std::equal(buffer, buffer + PREFIX_LEN, other.buffer)) {
                return false;
            }
            return *get_pointer() == *other.get_pointer();
        }
    }

    bool operator==(const char *str) const {
        size_t len = std::strlen(str);
        if (len != length)
            return false;

        if (!is_long()) {
            return std::strncmp(reinterpret_cast<const char *>(buffer), str, BUFFER_LEN) == 0;
        } else {
            if (!std::equal(buffer, buffer + PREFIX_LEN, str)) {
                return false;
            }
            std::string *ptr = get_pointer();
            if (ptr)
                return *ptr == std::string(str);
            return false;
        }
    }

    bool operator!=(const NewUmbra_t &other) const { return !(*this == other); }

    bool operator!=(const char *str) const { return !(*this == str); }

    bool operator<(const NewUmbra_t &other) const {
        int cmp;
        if (!is_long() && !other.is_long()) {
            cmp = std::memcmp(buffer, other.buffer, BUFFER_LEN);
        } else if (!is_long()) {
            cmp = std::memcmp(buffer, other.buffer, PREFIX_LEN);
            if (cmp == 0) {
                return this->to_string() < *(other.get_pointer());
            }
        } else if (!other.is_long()) {
            cmp = std::memcmp(buffer, other.buffer, PREFIX_LEN);
            if (cmp == 0) {
                return *(this->get_pointer()) < other.to_string();
            }
        } else {
            cmp = std::memcmp(buffer, other.buffer, PREFIX_LEN);
            if (cmp == 0) {
                std::string *ptr1 = get_pointer();
                std::string *ptr2 = other.get_pointer();
                return *ptr1 < *ptr2;
            }
        }
        if (cmp == 0) {
            return length < other.length;
        }
        return cmp < 0;
    }

    bool operator<(const char *str) const {
        uint32_t str_len = std::strlen(str);
        uint32_t min_length = std::min(length, str_len);
        int cmp;
        if (!is_long()) {
            cmp = std::memcmp(buffer, str, min_length);
        } else {
            cmp = std::memcmp(buffer, str, PREFIX_LEN);
            if (cmp == 0) {
                std::string *ptr1 = get_pointer();
                return *ptr1 < str;
            }
        }
        if (cmp == 0) {
            return length < str_len;
        }
        return cmp < 0;
    }

    bool operator>(const NewUmbra_t &other) const {
        int cmp;
        if (!is_long() && !other.is_long()) {
            cmp = std::memcmp(buffer, other.buffer, BUFFER_LEN);
        } else if (!is_long()) {
            cmp = std::memcmp(buffer, other.buffer, PREFIX_LEN);
            if (cmp == 0) {
                return this->to_string() > *(other.get_pointer());
            }
        } else if (!other.is_long()) {
            cmp = std::memcmp(buffer, other.buffer, PREFIX_LEN);
            if (cmp == 0) {
                return *(this->get_pointer()) > other.to_string();
            }
        } else {
            cmp = std::memcmp(buffer, other.buffer, PREFIX_LEN);
            if (cmp == 0) {
                std::string *ptr1 = get_pointer();
                std::string *ptr2 = other.get_pointer();
                return *ptr1 > *ptr2;
            }
        }
        if (cmp == 0) {
            return length > other.length;
        }
        return cmp > 0;
    }

    bool operator>(const char *str) const {
        uint32_t str_len = std::strlen(str);
        uint32_t min_length = std::min(length, str_len);
        int cmp;
        if (!is_long()) {
            cmp = std::memcmp(buffer, str, min_length);
        } else {
            cmp = std::memcmp(buffer, str, PREFIX_LEN);
            if (cmp == 0) {
                std::string *ptr1 = get_pointer();
                return *ptr1 > str;
            }
        }
        if (cmp == 0) {
            return length > str_len;
        }
        return cmp > 0;
    }

    // Concatenation operator
    NewUmbra_t operator+(const NewUmbra_t &rhs) const {
        std::string full_str = this->to_string() + rhs.to_string();
        return NewUmbra_t(full_str);
    }

    NewUmbra_t operator+(const char *rhs) const {
        std::string full_str = this->to_string() + std::string(rhs);
        return NewUmbra_t(full_str);
    }

    // Output Stream operator
    friend std::ostream &operator<<(std::ostream &os, const NewUmbra_t &str) {
        os << str.to_string();
        return os;
    }

    // method to get the string as c_str
    const char *get() const {
        if (!is_long()) {
            return reinterpret_cast<const char *>(buffer);
        } else {
            return get_pointer()->c_str();
        }
    }

    // Serialize method
    void serialize(std::vector<char> &outBuffer) const {

        outBuffer.insert(outBuffer.end(), reinterpret_cast<const char *>(&length),
                         reinterpret_cast<const char *>(&length) + sizeof(uint32_t));

        if (is_long()) {

            uintptr_t address = reinterpret_cast<uintptr_t>(get_pointer());
            outBuffer.insert(outBuffer.end(), reinterpret_cast<const char *>(&address),
                             reinterpret_cast<const char *>(&address) + sizeof(uintptr_t));
        } else {

            outBuffer.insert(outBuffer.end(), reinterpret_cast<const char *>(buffer),
                             reinterpret_cast<const char *>(buffer) + (BUFFER_LEN - PREFIX_LEN));
        }
    }

    // method to return std::string
    std::string to_string() const {
        if (is_long()) {
            std::string *ptr = get_pointer();
            if (ptr)
                return *ptr;
            return "";
        } else {
            return std::string(reinterpret_cast<const char *>(buffer), length);
        }
    }

    // compare method
    int compare(const NewUmbra_t &other) const { return to_string().compare(other.to_string()); }

    // Lowercase
    NewUmbra_t lower() const {
        NewUmbra_t result(*this);
        std::string lower_str = result.to_string();
        std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        result = NewUmbra_t(lower_str);
        return result;
    }

    // Uppercase
    NewUmbra_t upper() const {
        NewUmbra_t result(*this);
        std::string upper_str = result.to_string();
        std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(),
                       [](unsigned char c) { return std::toupper(c); });
        result = NewUmbra_t(upper_str);
        return result;
    }

    // method to set string
    void set(const char *str) {
        uint32_t new_length = static_cast<uint32_t>(std::strlen(str));
        if (is_long()) {
            delete get_pointer();
            clear_pointer();
        }
        length = new_length;
        if (new_length <= BUFFER_LEN) {
            std::copy(str, str + new_length, buffer);
            std::fill(buffer + new_length, buffer + BUFFER_LEN, static_cast<uint8_t>('\0'));
        } else {
            std::copy(str, str + PREFIX_LEN, buffer);
            std::string *ptr = new std::string(str);
            uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
            std::memcpy(buffer + PREFIX_LEN, &address, sizeof(uintptr_t));
        }
    }

    size_t size() const { return length; }
};

// Specialize std::hash for NewUmbra_t to use in unordered containers
namespace std {
template <> struct hash<NewUmbra_t> {
    size_t operator()(const NewUmbra_t &u) const { return std::hash<std::string>()(u.to_string()); }
};
} // namespace std