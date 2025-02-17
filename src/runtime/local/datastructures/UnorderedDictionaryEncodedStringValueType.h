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
 #include <vector>
 #include <unordered_map>
 #include <deque>
 #include <iostream>
 #include <mutex>
 
 /**
  * @brief A string value type that has a global unordered map that holds all the strings. 
  * Each string instance is simply a std::size_t. When creating a new string, 
  * the string is inserted into the dictionary and the index of the string is returned. 
  * For the String -> Index direction we have the string_index_map 
  * and for the Index -> String direction we have a deque.
  */
 
 struct UnorderedDictionaryEncodedString {
 private:
 
     // Global Vector (deque for thread safety) that Maps Index -> String
     inline static std::deque<std::string> index_string_vec;
 
     // Global Unordered Map that Maps String -> Index
     inline static std::unordered_map<std::string, std::size_t> string_index_map;
 
     // Mutex for thread safe execution
     inline static std::mutex dict_mutex;
 
     // Index for the given String in "index_string_vec"
     std::size_t index;
 
     /**
      * @brief Get the Index associated to the string. 
      * If it doesn't exist in the dictionary yet, it is first inserted.
      *
      * @param str The string to look up in the dictionary.
      * @return The index associated to str.
      */
     static std::size_t getIndex(const std::string &str) {
         // Aquire Lock
         std::lock_guard<std::mutex> lock(dict_mutex);
 
         auto search = string_index_map.find(str);
         if (search != string_index_map.end()) {
             // Element found
             return search->second;
         } 
         else {
             // Get Index for new string entry
             std::size_t newIndex = string_index_map.size();
 
             // Append string to end of vector
             index_string_vec.push_back(str);
 
             // Insert str into the Map
             string_index_map[str] = newIndex;
             return newIndex;
         }
     }
 
 public:
     // Default constructor: points to an empty string.
     UnorderedDictionaryEncodedString() : index(getIndex("")) {}
 
     // Constructor from a C-style string
     UnorderedDictionaryEncodedString(const char *str) : index(getIndex(str ? str : "")) {}
 
     // Constructor from a std::string
     UnorderedDictionaryEncodedString(const std::string &other) : index(getIndex(other)) {}
 
     // Copy constructor
     UnorderedDictionaryEncodedString(const UnorderedDictionaryEncodedString &other) : index(other.index) {}
 
     // Assignment operator
     UnorderedDictionaryEncodedString &operator=(const UnorderedDictionaryEncodedString &other) {
         if (this != &other) {
             index = other.index;
         }
         return *this;
     }
 
     // Equality operator compares by dictionary index
     bool operator==(const UnorderedDictionaryEncodedString &other) const {
         return index == other.index;
     }
 
     // Inequality operator
     bool operator!=(const UnorderedDictionaryEncodedString &other) const {
         return index != other.index;
     }
 
     // Overriding the Less than operator
     bool operator<(const UnorderedDictionaryEncodedString &other) const {
         return index_string_vec[index] < index_string_vec[other.index];
     }
 
 
     // Overriding the Greater than operator
     bool operator>(const UnorderedDictionaryEncodedString &other) const {
         return index_string_vec[index] > index_string_vec[other.index];
     }
 
     // Concatenation operator
     UnorderedDictionaryEncodedString operator+(const UnorderedDictionaryEncodedString &rhs) const {
         return UnorderedDictionaryEncodedString(index_string_vec[index] + index_string_vec[rhs.index]);
     }
 
     // Concatenation operator with C-style string
     UnorderedDictionaryEncodedString operator+(const char *rhs) const {
         std::string full_str = index_string_vec[index] + std::string(rhs);
         return UnorderedDictionaryEncodedString(full_str);
     }
 
     
     // Overload the output stream operator
     friend std::ostream &operator<<(std::ostream &os, const UnorderedDictionaryEncodedString &ds) {
         os << ds.to_string();
         return os;
     }
 
     // Size method
     size_t size() const
     {
         return index_string_vec[index].size();
     }
 
 
     // Set
     void set(const std::string &str) {
         index = getIndex(str);
     }
 
     // To String method
     std::string to_string() const {
         return index_string_vec[index];
     }
 
     //Convert to lowercase
     UnorderedDictionaryEncodedString lower() const {
         std::string str = index_string_vec[index];
         std::transform(str.begin(), str.end(), str.begin(), ::tolower);  // Convert to lowercase
         return UnorderedDictionaryEncodedString(str);
     }
 
     //Convert to uppercase
     UnorderedDictionaryEncodedString upper() const {
         std::string str = index_string_vec[index];
         std::transform(str.begin(), str.end(), str.begin(), ::toupper);  // Convert to lowercase
         return UnorderedDictionaryEncodedString(str);
     }
 
     
 };
 
 // Hash
 template <> struct std::hash<UnorderedDictionaryEncodedString> {
     std::size_t operator()(const UnorderedDictionaryEncodedString &key) const {
         // Compute the hash
         return std::hash<std::string>()(key.to_string());
     }
 };