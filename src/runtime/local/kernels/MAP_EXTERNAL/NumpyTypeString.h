#pragma once
#include <cstdint>

template <typename T>
struct NumpyTypeString;

template <>
struct NumpyTypeString<float> {
    static constexpr const char* value = "float32";
};

template <>
struct NumpyTypeString<double> {
    static constexpr const char* value = "float64";
};

template <>
struct NumpyTypeString<int64_t> {
    static constexpr const char* value = "int64";
};

template <>
struct NumpyTypeString<int32_t> {
    static constexpr const char* value = "int32";
};

template <>
struct NumpyTypeString<int8_t> {
    static constexpr const char* value = "int8";
};

template <>
struct NumpyTypeString<uint64_t> {
    static constexpr const char* value = "uint64";
};

template <>
struct NumpyTypeString<uint8_t> {
    static constexpr const char* value = "uint8";
};
template <>
struct NumpyTypeString<unsigned int> {
    static constexpr const char* value = "uint32";
};