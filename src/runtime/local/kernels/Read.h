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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_READ_H
#define SRC_RUNTIME_LOCAL_KERNELS_READ_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/ReadCsv.h>
#include <runtime/local/io/ReadMM.h>
#include <runtime/local/io/ReadParquet.h>
#include <runtime/local/io/ReadDaphne.h>
#include <parser/metadata/MetaDataParser.h>

#include <string>
#include <regex>
#include <map>

struct FileExt {
	static std::map<std::string, int> create_map() {
	std::map<std::string, int> m;
		m["csv"] = 0;
		m["mtx"] = 1;
		m["parquet"] = 2;
		m["dbdf"] = 3;
		return m;
	}
	static const std::map<std::string, int> map;
};

inline const std::map<std::string, int> FileExt::map = FileExt::create_map();

int extValue(const char * filename);

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes>
struct Read {
    static void apply(DTRes *& res, const char * filename, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
void read(DTRes *& res, const char * filename, DCTX(ctx)) {
    Read<DTRes>::apply(res, filename, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Read<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const char * filename, DCTX(ctx)) {

	FileMetaData fmd = MetaDataParser::readMetaData(filename);
	int extv = extValue(filename);
	switch(extv) {
	case 0:
		if(res == nullptr)
			res = DataObjectFactory::create<DenseMatrix<VT>>(
				fmd.numRows, fmd.numCols, false
			);
		readCsv(res, filename, fmd.numRows, fmd.numCols, ',');
		break;
	case 1:
		readMM(res, filename);
		break;
#ifdef USE_ARROW
	case 2:
		if(res == nullptr)
			res = DataObjectFactory::create<DenseMatrix<VT>>(
				fmd.numRows, fmd.numCols, false
			);
		readParquet(res, filename, fmd.numRows, fmd.numCols);
		break;
#endif
	case 3:
		readDaphne(res, filename);
                break;
        default:
            throw std::runtime_error("File extension not supported");
	}
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Read<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *& res, const char * filename, DCTX(ctx)) {

	FileMetaData fmd = MetaDataParser::readMetaData(filename);
	int extv = extValue(filename);
	switch(extv) {
	case 0:
		if(fmd.numNonZeros == -1)
                    throw std::runtime_error("Currently reading of sparse matrices requires a number of non zeros to be defined");

		if(res == nullptr)
			res = DataObjectFactory::create<CSRMatrix<VT>>(
				fmd.numRows, fmd.numCols, fmd.numNonZeros, false
			);

		// FIXME: ensure file is sorted, or set `sorted` argument correctly
		readCsv(res, filename, fmd.numRows, fmd.numCols, ',', fmd.numNonZeros, true);
		break;
	case 1:
		readMM(res, filename);
		break;
#ifdef USE_ARROW
	case 2:
		if(res == nullptr)
			res = DataObjectFactory::create<CSRMatrix<VT>>(
				fmd.numRows, fmd.numCols, fmd.numNonZeros, false
			);
		readParquet(res, filename,fmd.numRows, fmd.numCols,fmd.numNonZeros, false);
		break;
#endif
	case 3:
		readDaphne(res, filename);
                break;
        default:
            throw std::runtime_error("File extension not supported");
	}
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template<>
struct Read<Frame> {
    static void apply(Frame *& res, const char * filename, DCTX(ctx)) {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        
        ValueTypeCode * schema;
        if(fmd.isSingleValueType) {
            schema = new ValueTypeCode[fmd.numCols];
            for(size_t i = 0; i < fmd.numCols; i++)
                schema[i] = fmd.schema[0];
        }
        else
            schema = fmd.schema.data();
        
        std::string * labels;
        if(fmd.labels.empty())
            labels = nullptr;
        else
            labels = fmd.labels.data();
        
        if(res == nullptr)
            res = DataObjectFactory::create<Frame>(
                    fmd.numRows, fmd.numCols, schema, labels, false
            );
        
        readCsv(res, filename, fmd.numRows, fmd.numCols, ',', schema);
        
        if(fmd.isSingleValueType)
            delete[] schema;
    }
};


inline int extValue(const char * filename) {
	int extv;
	std::string fn(filename);
	auto pos = fn.find_last_of('.');
	std::string ext(fn.substr(pos+1)) ;
	if (FileExt::map.count(ext) > 0) {
		extv = FileExt::map.find(ext)->second;
	} else {
		extv = -1;
	}
	return extv;
}

#endif //SRC_RUNTIME_LOCAL_KERNELS_READ_H
