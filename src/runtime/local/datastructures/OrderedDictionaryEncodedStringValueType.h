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

 #include <string>
 #include <map>
 #include <mutex>
 #include <iostream>
 #include <algorithm>
 #include <cctype>
 #include <cstdint>
 
  /**
  * @brief A String value type that holds all the strings in a global ordered map. 
  * Each instance of a string has a pointer to its string and the lexicographical key. 
  * Its important to note that each unique string is only stored one time, 
  * meaning if two instances represent the same string they point to the same string, saving memory.
  * The lex_key can be used to perform comparisons (<, >, ...), because the generation of 
  * lex_key ensures that if std::string1 < std::string2 => lex_key1 < lex_kex2.
  * The first inserted string gets key 0 and all new strings are either placed above or below 0 
  * with a gap of 1000000000LL to ensure plenty of space in between for new insertions.
  */
 struct OrderedDictionaryEncodedString {
 private:
    
    inline static std::map<std::string, int64_t> string_order_map;
   
    inline static std::mutex dict_mutex;

   
    int64_t lex_key;
    const std::string* str_ptr;


    static void getOrInsert(const std::string &s, int64_t &key, const std::string* &ptr) {
        std::lock_guard<std::mutex> lock(dict_mutex);
        auto it = string_order_map.find(s);
        if (it != string_order_map.end()) {
            key = it->second;
            ptr = &(it->first);
        } else {
            
            const int64_t gap = 1000000000LL;
            int64_t new_key = 0;
            auto lower = string_order_map.lower_bound(s);
            if (string_order_map.empty()) {
                new_key = 0;
            } else if (lower == string_order_map.begin()) {
                
                new_key = lower->second - gap;
            } else if (lower == string_order_map.end()) {
               
                auto last = std::prev(string_order_map.end());
                new_key = last->second + gap;
            } else {
                
                auto pred = std::prev(lower);
                new_key = pred->second + (lower->second - pred->second) / 2;
                
                if (new_key == pred->second)
                    new_key = pred->second + 1;
            }
            auto insert_result = string_order_map.insert({s, new_key});
            key = insert_result.first->second;
            ptr = &(insert_result.first->first);
        }
    }

public:
    // Default constructor: represents the empty string.
    OrderedDictionaryEncodedString() {
        getOrInsert("", lex_key, str_ptr);
    }

    // Constructor from a C-style string.
    OrderedDictionaryEncodedString(const char *s) {
        std::string str(s ? s : "");
        getOrInsert(str, lex_key, str_ptr);
    }

    // Constructor from a std::string.
    OrderedDictionaryEncodedString(const std::string &s) {
        getOrInsert(s, lex_key, str_ptr);
    }

    // Copy constructor.
    OrderedDictionaryEncodedString(const OrderedDictionaryEncodedString &other)
        : lex_key(other.lex_key), str_ptr(other.str_ptr) {}

    // Assignment operator.
    OrderedDictionaryEncodedString &operator=(const OrderedDictionaryEncodedString &other) {
        if (this != &other) {
            lex_key = other.lex_key;
            str_ptr = other.str_ptr;
        }
        return *this;
    }

    // Comparios operators
    bool operator==(const OrderedDictionaryEncodedString &other) const {
        return lex_key == other.lex_key;
    }
    bool operator!=(const OrderedDictionaryEncodedString &other) const {
        return lex_key != other.lex_key;
    }
    bool operator<(const OrderedDictionaryEncodedString &other) const {
        return lex_key < other.lex_key;
    }
    bool operator>(const OrderedDictionaryEncodedString &other) const {
        return lex_key > other.lex_key;
    }
    bool operator<=(const OrderedDictionaryEncodedString &other) const {
        return lex_key <= other.lex_key;
    }
    bool operator>=(const OrderedDictionaryEncodedString &other) const {
        return lex_key >= other.lex_key;
    }

    // Concatenation operator
    OrderedDictionaryEncodedString operator+(const OrderedDictionaryEncodedString &rhs) const {
        return OrderedDictionaryEncodedString(*str_ptr + *rhs.str_ptr);
    }

    OrderedDictionaryEncodedString operator+(const char *rhs) const {
        return OrderedDictionaryEncodedString(*str_ptr + std::string(rhs));
    }

    // Overload the output stream operator
    friend std::ostream &operator<<(std::ostream &os, const OrderedDictionaryEncodedString &s) {
        os << *s.str_ptr;
        return os;
    }

    // Size method
    size_t size() const {
        return str_ptr->size();
    }

    // Set
    void set(const std::string &s) {
        getOrInsert(s, lex_key, str_ptr);
    }

    // To String method
    std::string to_string() const {
        return *str_ptr;
    }

    // Lowercase
    OrderedDictionaryEncodedString lower() const {
        std::string s = *str_ptr;
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return OrderedDictionaryEncodedString(s);
    }

    // Uppercase
    OrderedDictionaryEncodedString upper() const {
        std::string s = *str_ptr;
        std::transform(s.begin(), s.end(), s.begin(), ::toupper);
        return OrderedDictionaryEncodedString(s);
    }
};

// Specialization of std::hash for OrderedDictionaryEncodedString.
namespace std {
    template <>
    struct hash<OrderedDictionaryEncodedString> {
        std::size_t operator()(const OrderedDictionaryEncodedString &key) const {
            return std::hash<std::string>()(key.to_string());
        }
    };
}


