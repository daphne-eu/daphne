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
 
#include <tags.h>
#include <vector>
#include <cstdint>
#include <iostream>

#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/CheckEq.h>

#include <catch.hpp>

#include <duckdb.hpp>

TEST_CASE("DuckDB integration test", TAG_KERNELS) {
    std::cout << "" << std::endl;
    using namespace duckdb;
    
    DuckDB db(nullptr);
    Connection con(db);
    
    con.Query("CREATE TABLE integers(i INTEGER, j INTEGER)");
    
    con.Query("INSERT INTO integers VALUES (3, 4), (5, 6), (7, NULL)");

    auto result = con.Query("SELECT * FROM integers");
    if (!result->success) {
        std::cerr << result->error;
    }
}

