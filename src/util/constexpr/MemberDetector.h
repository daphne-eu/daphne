/*
 * Copyright 2023 The DAPHNE Consortium
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

#ifndef DAPHNE_MEMBERDETECTOR_H
#define DAPHNE_MEMBERDETECTOR_H

#include <type_traits>
#include <experimental/type_traits>

namespace daphne {
    /**
     * @brief Member detector following the [detector idiom](https://benjaminbrock.net/blog/detection_idiom.php)
     */
    namespace detector {

        template<typename T>
        using detect_member_method_apply = decltype(std::declval<T>().apply());

        template<typename T>
        using detect_static_method_apply = decltype(T::apply);

        /**
         * @brief Detects if a class has a member method called apply
         * 
         * @tparam T Class to check
         */
        template<typename T>
        inline constexpr
        bool has_member_method_apply_v = std::experimental::is_detected_v<detect_member_method_apply, T>;

        /**
         * @brief Detects if a class has a static method called apply
         * 
         * @tparam T Class to check
         */
        template<typename T>
        inline constexpr
        bool has_static_method_apply_v = std::experimental::is_detected_v<detect_static_method_apply, T>;

        /**
         * @brief Detects if a class has a non-static member method called apply
         * 
         * @tparam T Class to check
         */
        template<typename T>
        inline constexpr
        bool has_non_static_method_apply_v = has_member_method_apply_v<T> && !has_static_method_apply_v<T>;

    }
}

#endif //DAPHNE_MEMBERDETECTOR_H
