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


#ifndef DAPHNE_MLIR_SRC_UTIL_BITPUSHER_H
#define DAPHNE_MLIR_SRC_UTIL_BITPUSHER_H

#include <cstdint>

struct BitPusher {
    /**
     * @brief Test if the flag at the position is set and returns result
     */
    static bool isBitSet(int64_t flag, int64_t position){
        return ((flag >> position) & 1) == 1;
    }
    
    /**
     * @brief Sets the Flag at the given position with a value
     */
    static void setBit(int64_t& flag, int64_t position, int64_t val){
        val = !!val;
        flag ^= (-val ^ flag) & (0b1 << position);
    }
    
    /**
     * @brief Flips the bit of the Flag at the position
     */
    static void toggleBit(int64_t& flag, int64_t position){
        setBit(flag, position, !isBitSet(flag, position));
    }
};

#endif //DAPHNE_MLIR_SRC_UTIL_BITPUSHER_H
