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
    if (std::string(dbms) == "duckdb") {
        try {
            duckdb::DuckDB db(connection);
            duckdb::Connection con(db);
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
                case duckdb::LogicalTypeId::VARCHAR: {
                    auto* colData = DataObjectFactory::create<DenseMatrix<std::string>>(numRows, 1, false);
                    std::string* data = colData->getValues();
                    for (size_t row = 0; row < numRows; ++row) {
                        auto val = result->GetValue(col, row);
                        data[row] = val.IsNull() ? "" : val.ToString();
                    }
                    columns[col] = colData;
                    break;
                }
                case duckdb::LogicalTypeId::DATE: {
                    // For DATE columns, we convert the date to its string representation.
                    auto* colData = DataObjectFactory::create<DenseMatrix<std::string>>(numRows, 1, false);
                    std::string* data = colData->getValues();
                    for (size_t row = 0; row < numRows; ++row) {
                        auto val = result->GetValue(col, row);
                        data[row] = val.IsNull() ? "" : val.ToString();
                    }
                    columns[col] = colData;
                    break;
                }
                case duckdb::LogicalTypeId::DECIMAL: {
                    auto* colData = DataObjectFactory::create<DenseMatrix<double>>(numRows, 1, false);
                    double* data = colData->getValues();
                    for (size_t row = 0; row < numRows; ++row) {
                        auto val = result->GetValue(col, row);
                        data[row] = val.IsNull() ? 0.0 : val.GetValue<double>();
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
