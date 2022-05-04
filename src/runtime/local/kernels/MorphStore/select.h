/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/


#ifndef SRC_RUNTIME_LOCAL_KERNELS_MORPHSTORE_SELECT_H
#define SRC_RUNTIME_LOCAL_KERNELS_MORPHSTORE_SELECT_H

#include <cstdint>

enum class CompareOperation : uint32_t {
  Equal = 1,
  LessThan = 2,
  LessEqual = 3,
  GreaterThan = 4,
  GreaterEqual = 5,
  NotEqual = 6,
};

/// later use this CompareOperation Enum
//using mlir::daphne::CompareOperation;


template<class DTRes, class DTIn>
class Select {
  public:
    static void apply(DTRes * res, const DTIn * in, const char * inOn, CompareOperation cmp, uint64_t selValue) = delete;
};

template<class DTRes, class DTIn>
void select(DTRes * res, const DTIn * in, const char * inOn, CompareOperation cmp, uint64_t selValue) {
    Select<DTRes, DTIn>::apply(res, in, inOn, cmp, selValue);
}



#endif //SRC_RUNTIME_LOCAL_KERNELS_MORPHSTORE_SELECT_H
