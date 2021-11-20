/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <ir/daphneir/Daphne.h>

#include <string>
#include <vector>
#include <stdexcept>

namespace mlir::daphne
{
#include <ir/daphneir/DaphneDistributableOpInterface.cpp.inc>
}

using namespace mlir;

// ****************************************************************************
// Vector split and combine utility functions
// ****************************************************************************
// For families of operations.

template<class EwBinaryOp>
std::vector<mlir::Value> createEquivalentDistributedDAG_EwBinaryOp(EwBinaryOp *op, mlir::OpBuilder &builder,
                                                                   mlir::ValueRange distributedInputs)
{
    auto loc = op->getLoc();
    auto compute = builder.create<daphne::DistributedComputeOp>(loc,
        ArrayRef<Type>{daphne::HandleType::get(op->getContext(), op->getType())},
        distributedInputs);
    auto &block = compute.body().emplaceBlock();
    auto argLhs = block.addArgument(distributedInputs[0].getType().cast<daphne::HandleType>().getDataType());
    auto argRhs = block.addArgument(distributedInputs[1].getType().cast<daphne::HandleType>().getDataType());

    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(&block, block.begin());

        auto addOp = builder.create<EwBinaryOp>(loc, argLhs, argRhs);
        builder.create<daphne::ReturnOp>(loc, ArrayRef<Value>{addOp});
    }

    std::vector<Value> ret({builder.create<daphne::DistributedCollectOp>(loc, compute.getResult(0))});
    return ret;
}

template<class AggAllOp>
std::vector<mlir::Value> createEquivalentDistributedDAG_AggAllOp(AggAllOp *op, mlir::OpBuilder &builder,
                                                                   mlir::ValueRange distributedInputs)
{
    auto loc = op->getLoc();    
    auto compute = builder.create<daphne::DistributedComputeOp>(loc,
        ArrayRef<Type>{daphne::HandleType::get(op->getContext(), op->getType())},
        distributedInputs);
    auto &block = compute.body().emplaceBlock();
    auto argType = block.addArgument(distributedInputs[0].getType().cast<daphne::HandleType>().getDataType());    

    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(&block, block.begin());

        auto addOp = builder.create<AggAllOp>(loc, argType);
        builder.create<daphne::ReturnOp>(loc, ArrayRef<Value>{addOp});
    }

    std::vector<Value> ret({builder.create<daphne::DistributedCollectOp>(loc, compute.getResult(0))});
    return ret;
}

// ****************************************************************************
// CreateEquivalentDistributedDAG implementations
// ****************************************************************************

#define IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(OP) \
    std::vector<mlir::Value> mlir::daphne::OP::createEquivalentDistributedDAG(mlir::OpBuilder &builder, \
        mlir::ValueRange distributedInputs) \
    { \
        return createEquivalentDistributedDAG_EwBinaryOp(this, builder, distributedInputs); \
    }

#define IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_AGGALLOP(OP) \
    std::vector<mlir::Value> mlir::daphne::OP::createEquivalentDistributedDAG(mlir::OpBuilder &builder, \
        mlir::ValueRange distributedInputs) \
    { \
        return createEquivalentDistributedDAG_AggAllOp(this, builder, distributedInputs); \
    }


// Arithmetic
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwAddOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwSubOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwMulOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwDivOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwPowOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwModOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwLogOp)

// Min/max
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwMinOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwMaxOp)

// Logical
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwAndOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwOrOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwXorOp)

// Strings
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwConcatOp)

// Comparisons
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwEqOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwNeqOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwLtOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwLeOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwGtOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_EWBINARYOP(EwGeOp)

IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_AGGALLOP(AllAggSumOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_AGGALLOP(AllAggMinOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_AGGALLOP(AllAggMaxOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_AGGALLOP(AllAggMeanOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_AGGALLOP(AllAggVarOp)
IMPL_CREATEEQUIVALENTDISTRIBUTEDDAG_AGGALLOP(AllAggStddevOp)

