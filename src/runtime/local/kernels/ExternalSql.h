#ifndef SRC_RUNTIME_LOCAL_KERNELS_EXECUTEEXTERNALDBMS_H
#define SRC_RUNTIME_LOCAL_KERNELS_EXECUTEEXTERNALDBMS_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/kernels/CreateFrame.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <stdexcept>
#include <string>
#include <vector>
#include <cinttypes>

#include <duckdb.hpp>

#include <sql.h>
#include <sqlext.h>

#include <sqlite3.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

struct ExternalSql {
    // This function builds a DAPHNE Frame from the results of a DuckDB query.
    static void apply(Frame*& res, const char* query, const char* dbms,
                      const char* connection, DCTX(ctx));
};

// ****************************************************************************
// Implementation
// ****************************************************************************

void ExternalSql::apply(Frame*& res, const char* query, const char* dbms,
                        const char* connection, DCTX(ctx)) {
    if (std::string(query) == ""){
        throw std::runtime_error("Query is empty: please give a valid query.");
    }
    if (std::string(dbms) == "DuckDB" && std::string(connection) != "odbc") {
        try {
            // Establishing DuckDB connection
            duckdb::DuckDB db(connection);
            duckdb::Connection con(db);
            // Execute the query
            auto result = con.Query(query);
            if (!result || result->HasError()) {
                throw std::runtime_error("Query failed: " +
                                         (result ? result->GetError() : "Unknown error"));
            }

            const size_t numCols = result->ColumnCount();
            const size_t numRows = result->RowCount();

            // Prepare storage for DAPHNE column structures.
            std::vector<Structure*> columns(numCols);
            std::vector<const char*> colLabels(numCols);
            // Make sure the column label strings remain valid by storing them.
            std::vector<std::string> colLabelStorage;
            colLabelStorage.reserve(numCols);

            // Process each column according to its type.
            for (size_t col = 0; col < numCols; ++col) {
                const auto &duckType = result->types[col];
                const std::string& colName = result->names[col];
                colLabelStorage.push_back(colName);
                colLabels[col] = colLabelStorage.back().c_str();

                switch (duckType.id()) {
                case duckdb::LogicalTypeId::BOOLEAN: {
                    auto* colData = DataObjectFactory::create<DenseMatrix<int8_t>>(numRows, 1, false); // Use int8_t instead of bool
                    int8_t* data = colData->getValues();
                    for (size_t row = 0; row < numRows; ++row) {
                        auto val = result->GetValue(col, row);
                        data[row] = val.IsNull() ? 0 : (val.GetValue<bool>() ? 1 : 0); // Store 1 for true, 0 for false
                    }
                    columns[col] = colData;
                    break;
                }
                case duckdb::LogicalTypeId::TINYINT: {
                    auto* colData = DataObjectFactory::create<DenseMatrix<int8_t>>(numRows, 1, false);
                    int8_t* data = colData->getValues();
                    for (size_t row = 0; row < numRows; ++row) {
                        auto val = result->GetValue(col, row);
                        data[row] = val.IsNull() ? 0 : val.GetValue<int8_t>();
                    }
                    columns[col] = colData;
                    break;
                }
                case duckdb::LogicalTypeId::SMALLINT:
                case duckdb::LogicalTypeId::INTEGER: {
                    auto* colData = DataObjectFactory::create<DenseMatrix<int32_t>>(numRows, 1, false);
                    int32_t* data = colData->getValues();
                    for (size_t row = 0; row < numRows; ++row) {
                        auto val = result->GetValue(col, row);
                        data[row] = val.IsNull() ? 0 : val.GetValue<int32_t>();
                    }
                    columns[col] = colData;
                    break;
                }
                case duckdb::LogicalTypeId::HUGEINT:
                case duckdb::LogicalTypeId::BIGINT: {
                    auto* colData = DataObjectFactory::create<DenseMatrix<int64_t>>(numRows, 1, false);
                    int64_t* data = colData->getValues();
                    for (size_t row = 0; row < numRows; ++row) {
                        auto val = result->GetValue(col, row);
                        data[row] = val.IsNull() ? 0 : val.GetValue<int64_t>();
                    }
                    columns[col] = colData;
                    break;
                }
                case duckdb::LogicalTypeId::DECIMAL:
                case duckdb::LogicalTypeId::DOUBLE: {
                    auto* colData = DataObjectFactory::create<DenseMatrix<double>>(numRows, 1, false);
                    double* data = colData->getValues();
                    for (size_t row = 0; row < numRows; ++row) {
                        auto val = result->GetValue(col, row);
                        data[row] = val.IsNull() ? 0.0 : val.GetValue<double>();
                    }
                    columns[col] = colData;
                    break;
                }
                case duckdb::LogicalTypeId::DATE:
                case duckdb::LogicalTypeId::VARCHAR: {
                    auto* colData = DataObjectFactory::create<DenseMatrix<std::string>>(numRows, 1, false);
                    std::string* data = colData->getValues();
                    for (size_t row = 0; row < numRows; ++row) {
                        auto val = result->GetValue(col, row);
                        data[row] = val.ToString();
                    }
                    columns[col] = colData;
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported type: " + duckType.ToString());
                }
            }

            createFrame(res, columns.data(), numCols, colLabels.data(), numCols, ctx);

        } catch (const std::exception &e) {
            throw std::runtime_error("DuckDB Error: " + std::string(e.what()));
        }
    } else if (std::string(dbms) == "SQLite" && std::string(connection) != "odbc") {
        try {
            sqlite3* db = nullptr;
            sqlite3_stmt* stmt = nullptr;
            const char* remainingSql = query;  // Pointer to track remaining queries

            if (sqlite3_open(connection, &db) != SQLITE_OK) {
                throw std::runtime_error("Failed to open SQLite database: " + std::string(sqlite3_errmsg(db)));
            }

            while (remainingSql && *remainingSql) {
                // Prepare the next statement
                if (sqlite3_prepare_v2(db, remainingSql, -1, &stmt, &remainingSql) != SQLITE_OK) {
                    std::string errorMsg = sqlite3_errmsg(db);
                    sqlite3_close(db);
                    throw std::runtime_error("Failed to prepare SQLite query: " + errorMsg);
                }

                if (stmt == nullptr) {
                    continue;  // Skip empty statements
                }

                size_t numCols = sqlite3_column_count(stmt);

                // If the statement is not a SELECT (e.g., CREATE, DROP, INSERT)
                if (numCols == 0) {
                    // Create an empty frame as the result so that daphne does not give an error.
                    auto* noResultCol = DataObjectFactory::create<DenseMatrix<int64_t>>(1, 1, false);
                    noResultCol->getValues()[0] = 0;
                    Structure* colArray[1] = {noResultCol};
                    const char* colNames[1] = {"noResult"};
                    createFrame(res, colArray, 1, colNames, 1, ctx);

                    if (sqlite3_step(stmt) != SQLITE_DONE) {
                        std::string errorMsg = sqlite3_errmsg(db);
                        sqlite3_finalize(stmt);
                        sqlite3_close(db);
                        throw std::runtime_error("SQLite execution error: " + errorMsg);
                    }
                    sqlite3_finalize(stmt);
                    continue;  // Move to the next statement
                }

                // SELECT Statements: Set up column storage
                std::vector<Structure *> columns(numCols);
                std::vector<const char *> colLabels(numCols);
                std::vector<std::string> colLabelStorage(numCols);

                for (size_t col = 0; col < numCols; ++col) {
                    colLabelStorage[col] = sqlite3_column_name(stmt, col);
                    colLabels[col] = colLabelStorage[col].c_str();
                }

                std::vector<std::vector<std::string>> rowData(numCols);
                std::vector<int> colTypes(numCols);
                size_t numRows = 0;

                // Fetch rows
                while (sqlite3_step(stmt) == SQLITE_ROW) {
                    for (size_t col = 0; col < numCols; ++col) {
                        int colType = sqlite3_column_type(stmt, col);

                        // Make sure that NULL does not change the column type in case it came last
                        if(colType != SQLITE_NULL || colTypes[col] == 0){
                        colTypes[col] = colType;
                        }

                        switch (colType) {
                        case SQLITE_INTEGER:
                            rowData[col].push_back(std::to_string(sqlite3_column_int64(stmt, col)));
                            break;
                        case SQLITE_FLOAT:
                            rowData[col].push_back(std::to_string(sqlite3_column_double(stmt, col)));
                            break;
                        case SQLITE_TEXT:
                            rowData[col].push_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, col)));
                            break;
                        case SQLITE_BLOB: {
                            const void* blobData = sqlite3_column_blob(stmt, col);
                            int blobSize = sqlite3_column_bytes(stmt, col);
                            std::string binaryData(static_cast<const char*>(blobData), blobSize);
                            rowData[col].push_back(binaryData);
                            break;
                        }
                        case SQLITE_NULL:
                            rowData[col].push_back("");  // Store empty string for NULL values
                            break;
                        default:
                            throw std::runtime_error("Unsupported column type in SQLite result processing");
                        }
                    }
                    numRows++;
                }

                sqlite3_finalize(stmt);

                if(numRows == 0){
                    auto* noResultCol = DataObjectFactory::create<DenseMatrix<int64_t>>(1, 1, false);
                    noResultCol->getValues()[0] = 0;
                    Structure* colArray[1] = {noResultCol};
                    const char* colNames[1] = {"emptyTable"};
                    createFrame(res, colArray, 1, colNames, 1, ctx);
                    continue;
                }

                // Convert results to DAPHNE Frame
                for (size_t col = 0; col < numCols; ++col) {
                    if (!rowData[col].empty()) {
                        switch (colTypes[col]) {
                        case SQLITE_INTEGER: {
                            auto* colData = DataObjectFactory::create<DenseMatrix<int64_t>>(numRows, 1, false);
                            int64_t* data = colData->getValues();
                            for (size_t row = 0; row < numRows; ++row) {
                                data[row] = rowData[col][row].empty() ? 0 : std::stoll(rowData[col][row]);
                            }
                            columns[col] = colData;
                            break;
                        }
                        case SQLITE_FLOAT: {
                            auto* colData = DataObjectFactory::create<DenseMatrix<double>>(numRows, 1, false);
                            double* data = colData->getValues();
                            for (size_t row = 0; row < numRows; ++row) {
                                data[row] = rowData[col][row].empty() ? 0.0 : std::stod(rowData[col][row]);
                            }
                            columns[col] = colData;
                            break;
                        }
                        case SQLITE_TEXT:
                        case SQLITE_BLOB:
                        case SQLITE_NULL: {
                            auto* colData = DataObjectFactory::create<DenseMatrix<std::string>>(numRows, 1, false);
                            std::string* data = colData->getValues();
                            for (size_t row = 0; row < numRows; ++row) {
                                data[row] = rowData[col][row].empty() ? "NULL" : rowData[col][row];
                            }
                            columns[col] = colData;
                            break;
                        }
                        default:
                            throw std::runtime_error("Unsupported column type in SQLite result processing");
                        }
                    }
                }

                if (!columns.empty()) {
                    createFrame(res, columns.data(), numCols, colLabels.data(), numCols, ctx);
                } else {
                    auto* noResultCol = DataObjectFactory::create<DenseMatrix<int64_t>>(1, 1, false);
                    noResultCol->getValues()[0] = 0;
                    Structure* colArray[1] = {noResultCol};
                    const char* colNames[1] = {"noResult"};
                    createFrame(res, colArray, 1, colNames, 1, ctx);
                }

            }

            sqlite3_close(db);
        } catch (const std::exception &e) {
            throw std::runtime_error("SQLite Error: " + std::string(e.what()));
        }
    } else if (std::string(connection) == "odbc") {
        try {
            SQLHANDLE env;
            SQLHANDLE dbc;
            SQLRETURN ret;

            ret = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &env);
            if (!SQL_SUCCEEDED(ret)) {
                throw std::runtime_error("ODBC: Failed to allocate environment handle.");
            }

            ret = SQLSetEnvAttr(env, SQL_ATTR_ODBC_VERSION, (void*)SQL_OV_ODBC3, 0);
            if (!SQL_SUCCEEDED(ret)) {
                SQLFreeHandle(SQL_HANDLE_ENV, env);
                throw std::runtime_error("ODBC: Failed to set environment attribute.");
            }

            ret = SQLAllocHandle(SQL_HANDLE_DBC, env, &dbc);
            if (!SQL_SUCCEEDED(ret)) {
                SQLFreeHandle(SQL_HANDLE_ENV, env);
                throw std::runtime_error("ODBC: Failed to allocate connection handle.");
            }


            //TODO: solve the memory error when using a string to connect
            std::string dsn = std::string(dbms);
            ret = SQLConnect(dbc, reinterpret_cast<SQLCHAR*>(dsn.data()), SQL_NTS, NULL, 0, NULL, 0);
            if (!SQL_SUCCEEDED(ret)) {
                SQLCHAR sqlState[6] = {0}, message[256] = {0};
                SQLINTEGER nativeError;
                SQLSMALLINT textLength;
                SQLGetDiagRec(SQL_HANDLE_DBC, dbc, 1, sqlState, &nativeError,
                              message, sizeof(message), &textLength);
                SQLFreeHandle(SQL_HANDLE_DBC, dbc);
                SQLFreeHandle(SQL_HANDLE_ENV, env);
                throw std::runtime_error("ODBC Connection Failed: " + std::string((char*)message));
            }

            // Allocate a statement handle.
            SQLHANDLE stmt;
            ret = SQLAllocHandle(SQL_HANDLE_STMT, dbc, &stmt);
            if (!SQL_SUCCEEDED(ret)) {
                SQLDisconnect(dbc);
                SQLFreeHandle(SQL_HANDLE_DBC, dbc);
                SQLFreeHandle(SQL_HANDLE_ENV, env);
                throw std::runtime_error("ODBC: Failed to allocate statement handle.");
            }

            // Convert the query into a mutable SQLCHAR buffer.
            std::string sqlQuery(query);
            std::vector<char> queryBuffer(sqlQuery.begin(), sqlQuery.end());
            queryBuffer.push_back('\0');

            ret = SQLExecDirect(stmt, (SQLCHAR*)queryBuffer.data(), SQL_NTS);
            if (!SQL_SUCCEEDED(ret)) {
                SQLCHAR sqlState[6] = {0}, message[256] = {0};
                SQLINTEGER nativeError;
                SQLSMALLINT textLength;
                SQLGetDiagRec(SQL_HANDLE_STMT, stmt, 1, sqlState, &nativeError,
                              message, sizeof(message), &textLength);
                SQLFreeHandle(SQL_HANDLE_STMT, stmt);
                SQLDisconnect(dbc);
                SQLFreeHandle(SQL_HANDLE_DBC, dbc);
                SQLFreeHandle(SQL_HANDLE_ENV, env);
                throw std::runtime_error("ODBC Query Failed: " + std::string((char*)message));
            }

            // Retrieve the number of result columns.
            SQLSMALLINT numCols;
            ret = SQLNumResultCols(stmt, &numCols);
            if (!SQL_SUCCEEDED(ret) || numCols <= 0) {
                SQLFreeHandle(SQL_HANDLE_STMT, stmt);
                SQLDisconnect(dbc);
                SQLFreeHandle(SQL_HANDLE_DBC, dbc);
                SQLFreeHandle(SQL_HANDLE_ENV, env);
                throw std::runtime_error("ODBC: No result columns or error retrieving columns.");
            }

            // Retrieve column names and detect column types.
            std::vector<std::string> colLabelStorage(numCols);
            std::vector<const char*> colLabels(numCols);
            std::vector<SQLSMALLINT> colTypes(numCols);
            for (SQLSMALLINT col = 0; col < numCols; ++col) {
                SQLCHAR colNameBuf[256];
                SQLSMALLINT nameLen;
                SQLSMALLINT dataType;
                ret = SQLDescribeCol(stmt, col + 1, colNameBuf, sizeof(colNameBuf),
                                     &nameLen, &dataType, nullptr, nullptr, nullptr);
                if (SQL_SUCCEEDED(ret)) {
                    colLabelStorage[col] = std::string((char*)colNameBuf);
                    colTypes[col] = dataType;
                } else {
                    colLabelStorage[col] = "col" + std::to_string(col);
                    colTypes[col] = SQL_CHAR; // default to string if unknown
                }
                colLabels[col] = colLabelStorage[col].c_str();
            }

            // Prepare containers for each column type.
            // (Each column will use only one of these based on its detected type)
            std::vector<std::vector<int32_t>> intColumns(numCols);
            std::vector<std::vector<int64_t>> bigintColumns(numCols);
            std::vector<std::vector<double>> doubleColumns(numCols);
            std::vector<std::vector<std::string>> stringColumns(numCols);
            std::vector<Structure*> columns(numCols);

            // Fetch the rows.
            while (true) {
                ret = SQLFetch(stmt);
                if (ret == SQL_NO_DATA){
                    break;
                }
                if (!SQL_SUCCEEDED(ret)) {
                    throw std::runtime_error("ODBC: Error during SQLFetch.");
                }
                for (SQLSMALLINT col = 0; col < numCols; ++col) {
                    SQLLEN indicator = 0;
                    switch (colTypes[col]) {
                    case SQL_INTEGER:
                    case SQL_SMALLINT:
                    case SQL_TINYINT: {
                        SQLINTEGER intVal = 0;
                        ret = SQLGetData(stmt, col + 1, SQL_C_SLONG, &intVal, sizeof(intVal), &indicator);
                        if (SQL_SUCCEEDED(ret) && indicator != SQL_NULL_DATA) {
                            intColumns[col].push_back(intVal);
                        } else {
                            intColumns[col].push_back(0);
                        }
                        break;
                    }
                    case SQL_BIGINT: {
                        SQLBIGINT bigintVal = 0;
                        ret = SQLGetData(stmt, col + 1, SQL_C_SBIGINT, &bigintVal, sizeof(bigintVal), &indicator);
                        if (SQL_SUCCEEDED(ret) && indicator != SQL_NULL_DATA) {
                            bigintColumns[col].push_back(bigintVal);
                        } else {
                            bigintColumns[col].push_back(0);
                        }
                        break;
                    }
                    case SQL_DOUBLE:
                    case SQL_FLOAT:
                    case SQL_REAL:
                    case SQL_DECIMAL:
                    case SQL_NUMERIC: {
                        double dblVal = 0;
                        ret = SQLGetData(stmt, col + 1, SQL_C_DOUBLE, &dblVal, sizeof(dblVal), &indicator);
                        if (SQL_SUCCEEDED(ret) && indicator != SQL_NULL_DATA) {
                            doubleColumns[col].push_back(dblVal);
                        } else {
                            doubleColumns[col].push_back(0.0);
                        }
                        break;
                    }
                    default: {
                        char buf[10000] = {0};
                        // Adjust BufferLength to leave space for null terminator
                        ret = SQLGetData(stmt, col + 1, SQL_C_CHAR, buf, sizeof(buf) - 1, &indicator);
                        // Explicitly null-terminate the buffer
                        buf[sizeof(buf) - 1] = '\0';
                        if (SQL_SUCCEEDED(ret) && indicator != SQL_NULL_DATA) {
                            stringColumns[col].push_back(std::string(buf));
                        } else {
                            stringColumns[col].push_back("");
                        }
                        break;
                    }
                    }
                }
            }


            for (SQLSMALLINT col = 0; col < numCols; ++col) {
                if (colTypes[col] == SQL_INTEGER || colTypes[col] == SQL_SMALLINT || colTypes[col] == SQL_TINYINT) {
                    size_t rows = intColumns[col].size();
                    auto* colData = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, 1, false);
                    int32_t* data = colData->getValues();
                    for (size_t row = 0; row < rows; ++row)
                        data[row] = intColumns[col][row];
                    columns[col] = colData;
                }
                else if (colTypes[col] == SQL_BIGINT) {
                    size_t rows = bigintColumns[col].size();
                    auto* colData = DataObjectFactory::create<DenseMatrix<int64_t>>(rows, 1, false);
                    int64_t* data = colData->getValues();
                    for (size_t row = 0; row < rows; ++row)
                        data[row] = bigintColumns[col][row];
                    columns[col] = colData;
                }
                else if (colTypes[col] == SQL_DOUBLE || colTypes[col] == SQL_FLOAT ||
                         colTypes[col] == SQL_REAL || colTypes[col] == SQL_DECIMAL || colTypes[col] == SQL_NUMERIC) {
                    size_t rows = doubleColumns[col].size();
                    auto* colData = DataObjectFactory::create<DenseMatrix<double>>(rows, 1, false);
                    double* data = colData->getValues();
                    for (size_t row = 0; row < rows; ++row)
                        data[row] = doubleColumns[col][row];
                    columns[col] = colData;
                }
                else {
                    size_t rows = stringColumns[col].size();
                    auto* colData = DataObjectFactory::create<DenseMatrix<std::string>>(rows, 1, false);
                    std::string* data = colData->getValues();
                    for (size_t row = 0; row < rows; ++row)
                        data[row] = stringColumns[col][row];
                    columns[col] = colData;
                }
            }
            createFrame(res, columns.data(), numCols, colLabels.data(), numCols, ctx);

            // Cleanup ODBC handles.
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
            SQLDisconnect(dbc);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
        }
        catch (const std::exception &e) {
            throw std::runtime_error("ODBC Error: " + std::string(e.what()));
        }
    } else {
        throw std::runtime_error("Unsupported DBMS: " + std::string(dbms));
    }
}

// ****************************************************************************
// Convenience function
// ****************************************************************************

void externalSql(Frame*& res, const char* query, const char* dbms,
                 const char* connection, DCTX(ctx)) {
    ExternalSql::apply(res, query, dbms, connection, ctx);
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_EXECUTEEXTERNALDBMS_H
