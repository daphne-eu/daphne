/*
 * Copyright 2021 The DAPHNE Consortium
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

#ifndef DAPHNE_PROTOTYPE_SRC_UTIL_DEDUCETYPE_H
#define DAPHNE_PROTOTYPE_SRC_UTIL_DEDUCETYPE_H

#include <type_traits>
#include <utility>
#include <runtime/local/datastructures/ValueTypeCode.h>

/// @todo move somewhere better suited
template<typename T>
struct is_ValueTypeCode : std::false_type {};

template<>
struct is_ValueTypeCode<ValueTypeCode> : std::true_type {};
template<>
struct is_ValueTypeCode<const ValueTypeCode> : std::true_type {};
template<>
struct is_ValueTypeCode<ValueTypeCode&> : std::true_type {};
template<>
struct is_ValueTypeCode<const ValueTypeCode&> : std::true_type {};

template<typename T>
inline constexpr bool is_ValueTypeCode_v = is_ValueTypeCode<T>::value;

/**
 * @brief Helper class to handle ValueTypeCode to C++ data type mapping.
 *
 * This class is designed to ease the handling of types of frames.
 * As first template parameter, it takes a templated class with an static <u>apply</u> function, which is later called
 * with the deduced data types.
 *
 * \n\n<b>Usage:</b>\n
 * <u>1. one type</u>
 * <p>DeduceValueTypeAndExecute\<Kernel\>::apply(ValueTypeCode::SI32, 42);\n
 * ==> calls Kernel\<int32_t\>::apply(42)</p>
 *
 * <p>DeduceValueTypeAndExecute\<Kernel\>::apply(ValueTypeCode::SI32, args...);\n
 * ==> calls Kernel\<int32_t\>::apply(args...)</p>
 *
 *  <p>DeduceValueTypeAndExecute\<Kernel, Foo, Bar\>::apply(ValueTypeCode::SI32, 42);\n
 *  ==> calls Kernel\<Foo, Bar, int32_t\>::apply(42)</p>
 *
 * <u>2. multiple types</u>
 *
 * <p>DeduceValueTypeAndExecute\<Kernel\>::apply(ValueTypeCode::SI32, ValueTypeCode::UI64,
 * ValueTypeCode::F64, args...);\n
 * ==> calls Kernel\<int32_t, uint64_t, double\>::apply(args...)</p>
 *
 * <p>DeduceValueTypeAndExecute\<Kernel, Foo, Bar\>::apply(ValueTypeCode::SI32, ValueTypeCode::UI64,
 * ValueTypeCode::F64, args...);\n
 * ==> calls Kernel\<Foo, Bar, int32_t, uint64_t, double\>::apply(args...)</p>
 *
 * <b>Warning:</b> (|ValueTypeCode| ^ Levels) switch-case-branches are created, where Levels is the amount of
 * ValueTypeCode arguments. Use with care.
 *
 *
 * @tparam TExec Executable - Templated class with static apply function.
 * @tparam TList Additional template parameters passed to the Executable.
 */
template < template < typename ... > typename TExec, typename ... TList >
class DeduceValueTypeAndExecute;

template < uint64_t depth, template <typename ... > typename TExec, typename...TList>
class DeduceValueType_Helper {
    template < template < typename ... > typename, typename ... >
    friend class DeduceValueTypeAndExecute;
    template< uint64_t, template <typename ... > typename, typename ... >
    friend class DeduceValueType_Helper;
    
    template < typename ... TArgs >
    static void apply(ValueTypeCode vtc, TArgs&&...args){
        if constexpr (depth > 1){
            switch (vtc) {
                case ValueTypeCode::SI8:
                    DeduceValueType_Helper<depth - 1, TExec, TList..., int8_t  >::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::SI32:
                    DeduceValueType_Helper<depth - 1, TExec, TList..., int32_t >::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::SI64:
                    DeduceValueType_Helper<depth - 1, TExec, TList..., int64_t >::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::UI8:
                    DeduceValueType_Helper<depth - 1, TExec, TList..., uint8_t >::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::UI32:
                    DeduceValueType_Helper<depth - 1, TExec, TList..., uint32_t>::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::UI64:
                    DeduceValueType_Helper<depth - 1, TExec, TList..., uint64_t>::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::F32:
                    DeduceValueType_Helper<depth - 1, TExec, TList..., float   >::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::F64:
                    DeduceValueType_Helper<depth - 1, TExec, TList..., double  >::apply(std::forward<TArgs>(args)...); return;
                default:
                    throw std::runtime_error("DeduceValueType_Helper::apply: unknown value type code");
            }
        } else {
            switch (vtc) {
                case ValueTypeCode::SI8:
                    TExec<TList..., int8_t  >::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::SI32:
                    TExec<TList..., int32_t >::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::SI64:
                    TExec<TList..., int64_t >::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::UI8:
                    TExec<TList..., uint8_t >::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::UI32:
                    TExec<TList..., uint32_t>::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::UI64:
                    TExec<TList..., uint64_t>::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::F32:
                    TExec<TList..., float   >::apply(std::forward<TArgs>(args)...); return;
                case ValueTypeCode::F64:
                    TExec<TList..., double  >::apply(std::forward<TArgs>(args)...); return;
                default:
                    throw std::runtime_error("DeduceValueType_Helper::apply: unknown value type code");
            }
        }
    
    }
};


template < template < typename ... > typename TExec, typename ... TList >
class DeduceValueTypeAndExecute {
    /**
     * Count how many types in TArgs are of type ValueTypeCode
     * @tparam TArgs Types to test and count
     */
    template<typename...TArgs>
    struct count_vtc {
        /// Count of ValueTypeCodes in TArgs
        static const uint64_t count = 0 + ( ... + (is_ValueTypeCode_v<TArgs> ? 1 : 0));
    };
  public:
    /**
     * @brief Calls the executable after deducing the value types.
     *
     * The ValueTypeCodes to deduce have to be clustered as the first arguments.
     * @tparam Tvtc
     * @param vtc
     */
    template <typename...Tvtc>
    static void apply(Tvtc&& ... vtc){
        DeduceValueType_Helper<count_vtc<Tvtc...>::count, TExec, TList...>::apply(std::forward<Tvtc>(vtc)...);
    }
};



#endif //DAPHNE_PROTOTYPE_SRC_UTIL_DEDUCETYPE_H
