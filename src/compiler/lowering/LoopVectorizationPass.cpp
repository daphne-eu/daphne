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

#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>
#include <mlir/dialect/SCF/IR/SCF.h>

using namespace mlir;

/**
 * @brief Replace loops with indexed matrix accesses by
 * their corresponding element wise matrix operations whenever possible.
 */
class LoopVectorizationPass
    : public PassWrapper<LoopVectorizationPass, OperationPass<ModuleOp>> {
  public:
    LoopVectorizationPass() {}

    void runOnOperation() final;

    StringRef getArgument() const final { return "loop-vectorization"; }
    StringRef getDescription() const final {
        return "Replace loops with indexed matrix accesses by their "
               "corresponding element wise matrix operations whenever possible";
    }
};

// Helper functions

// Check if a value is a constant integer with the value i
bool isConstantInteger(mlir::Value value, int i) {
    if (auto op = dyn_cast<daphne::ConstantOp>(value.getDefiningOp())) {
        auto val = op.getValue();
        if (val.isa<IntegerAttr>() &&
            val.dyn_cast<IntegerAttr>().getValue() == i) {
            return true;
        }
    }

    return false;
}

// Check if a loop returns only matrices of the same dimensions
// If yes, return the dimensions of the matrices
std::optional<std::pair<long int, long int>>
getHomogenousReturnSize(scf::ForOp &loop) {
    long int rows = 0;
    long int cols = 0;

    for (auto resType : loop.getResultTypes()) {
        // Result must be a matrix type to even be considered
        if (auto res = dyn_cast<daphne::MatrixType>(resType)) {
            // Since Matrix types always have a size with cols, rows > 0, we can
            // simply initialize them with 0 and then set them to the first
            // value we encounter
            if (rows == 0) {
                rows = res.getNumRows();
            }
            if (cols == 0) {
                cols = res.getNumCols();
            }

            // Ensure that all the matrix types returned have exactly the same
            // size
            if (rows != res.getNumRows() || cols != res.getNumCols()) {
                return std::nullopt;
            }
        } else {
            return std::nullopt;
        }
    }

    return std::make_pair(rows, cols);
}

// Indices are generally not referenced directly in the loop body, but rather by
// a sequence of multiplications by 1 and casts. This function resolves these.
mlir::Value resolveIndex(mlir::Value val) {
    if (val.isa<BlockArgument>()) {
        return val;
    }

    auto def = val.getDefiningOp();
    if (auto castOp = dyn_cast<daphne::CastOp>(def)) {
        return resolveIndex(castOp.getOperand());
    } else if (auto mulOp = dyn_cast<daphne::EwMulOp>(def)) {
        if (isConstantInteger(mulOp.getOperand(1), 1)) {
            return resolveIndex(mulOp.getOperand(0));
        } else {
            return val;
        }
    } else {
        return val;
    }
}

// Check if a loop has no side effects (we can only vectorize such loops)
bool loopHasNoSideEffects(scf::ForOp &loopOp) {
    bool result = true;

    loopOp.getBody()->walk([&](mlir::Operation *op) {
        if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
            if (!memInterface.hasNoEffect()) {
                result = false;
            }
        } else {
            // Some ops don't implement the MemoryEffectOpInterface but are
            // still treated as being side-effect-less for our purposes.
            if (!dyn_cast<daphne::SliceColOp>(op) &&
                !dyn_cast<daphne::SliceRowOp>(op) &&
                !dyn_cast<daphne::InsertColOp>(op) &&
                !dyn_cast<daphne::InsertRowOp>(op) &&
                !dyn_cast<daphne::CastOp>(op)) {
                result = false;
            }
        }
    });

    return result;
}

// An InnerLoop represents a loop whose body we are trying to vectorize
class InnerLoop : public scf::ForOp {
  public:
    // Try to create an InnerLoop from an outer loop
    static std::optional<InnerLoop>
    getCompatibleInnerLoop(scf::ForOp &outerLoop, mlir::Location loc,
                           OpBuilder &builder) {
        auto returnSize = getHomogenousReturnSize(outerLoop);

        if (!returnSize.has_value()) {
            return std::nullopt;
        }

        auto [outerRows, outerCols] = *returnSize;

        // Can only optimize with step sizes of 1 and lower bounds of 0
        if (!isConstantInteger(outerLoop.getStep(), 1) ||
            !isConstantInteger(outerLoop.getLowerBound(), 0)) {
            return std::nullopt;
        }

        auto &body = outerLoop.getBody()->getOperations();
        // Body must contain exactly 4 ops: CastOp, EwMulOp, ForOp, YieldOp
        if (body.size() != 4) {
            return std::nullopt;
        }

        // First Op must be a CastOp
        auto it = body.begin();
        if (!dyn_cast<daphne::CastOp>(it)) {
            return std::nullopt;
        }

        // Second Op must be EwMulOp
        it++;
        if (!dyn_cast<daphne::EwMulOp>(&(*it))) {
            return std::nullopt;
        }

        it++;
        // Third Op is the actual inner loop
        if (auto innerLoop = dyn_cast<scf::ForOp>(&(*it))) {
            auto innerReturnSize = getHomogenousReturnSize(innerLoop);

            if (!loopHasNoSideEffects(
                    innerLoop) || // Must not have any side effects
                !innerReturnSize.has_value()) {
                return std::nullopt;
            }

            auto [innerRows, innerCols] = *innerReturnSize;

            if (!isConstantInteger(innerLoop.getStep(), 1) ||
                !isConstantInteger(innerLoop.getLowerBound(), 0)) {
                return std::nullopt;
            }

            // inner and outer dimensions and number of results must match
            if (innerRows == outerRows && innerCols == outerCols &&
                innerLoop.getNumResults() == outerLoop.getNumResults()) {
                // determine if i refers to rows or cols
                if (isConstantInteger(innerLoop.getUpperBound(), outerCols) &&
                    isConstantInteger(outerLoop.getUpperBound(), outerRows)) {
                    return InnerLoop(
                        innerLoop, outerRows, outerCols,
                        outerLoop
                            .getInductionVar(), // outerLoop var is rows (i) and
                        innerLoop
                            .getInductionVar(), // innerLoop var is cols (j)
                        loc, builder);
                }
                if (isConstantInteger(outerLoop.getUpperBound(), outerCols) &&
                    isConstantInteger(innerLoop.getUpperBound(), outerRows)) {
                    return InnerLoop(
                        innerLoop, outerRows, outerCols,
                        innerLoop
                            .getInductionVar(), // innerLoop var is rows (i) and
                        outerLoop
                            .getInductionVar(), // outerLoop var is cols (j)
                        loc, builder);
                }
            }
        }

        return std::nullopt;
    }

    // If vectorization fails, we must delete all the extra values we created
    void flushBuiltVals() {
        for (auto it = builtVals.rbegin(); it != builtVals.rend(); it++) {
            it->getDefiningOp()->erase();
        }
        builtVals.clear();
    }

    // Delete all the unnecessary casts we inserted.
    // This naive implementation is safe because we only insert trivial casts
    void flushBuiltCasts() {
        for (auto val : builtVals) {
            if (auto castOp = dyn_cast<daphne::CastOp>(val.getDefiningOp())) {
                castOp.getResult().replaceAllUsesWith(castOp.getOperand());
                castOp.erase();
            }
        }
    }

    // Check whether an operation inserts a value into a matrix of the correct
    // size at position i, j and return the value that is inserted if so.
    std::optional<mlir::Value> ijInsert(mlir::Operation *op) {
        if (auto insertRowOp = dyn_cast<daphne::InsertRowOp>(op)) {
            auto insertColOp = static_cast<daphne::InsertColOp>(
                insertRowOp.getOperand(1)
                    .getDefiningOp()); // Accesses always happen this way, so
                                       // static_cast is fine.
            auto resultType = static_cast<daphne::MatrixType>(
                insertRowOp.getResult().getType().cast<daphne::MatrixType>());

            if (resultType.getNumRows() != rows ||
                resultType.getNumCols() != cols) {
                return std::nullopt;
            }

            // Make sure the insertion indices are correct
            if (isi(insertRowOp.getOperand(2)) &&
                isj(insertColOp.getOperand(2)) &&
                isiplus1(insertRowOp.getOperand(3)) &&
                isjplus1(insertColOp.getOperand(3))) {
                return insertColOp.getOperand(1);
            }
        }
        return std::nullopt;
    }

    std::optional<mlir::Value> buildVectorized(mlir::Operation *op) {
        if (auto sliceColOp = dyn_cast<daphne::SliceColOp>(*op)) {
            if (auto access = ijAccess(sliceColOp)) {
                auto built_val =
                    builder
                        .create<daphne::CastOp>(loc, access->getType(),
                                                *access)
                        .getResult(); // insert an empty cast. Will be removed
                                      // at the end of the pass. Necessary
                                      // to move values outside of the loop.
                builtVals.push_back(built_val);
                return built_val;
            }
        } else if (auto constantOp = dyn_cast<daphne::ConstantOp>(*op)) {
            return constantOp.getResult();
        } else if (auto ewAddOp = dyn_cast<daphne::EwAddOp>(*op)) {
            return vectorizeBinOp(ewAddOp);
        } else if (auto ewMulOp = dyn_cast<daphne::EwMulOp>(*op)) {
            return vectorizeBinOp(ewMulOp);
        } else if (auto ewSubOp = dyn_cast<daphne::EwSubOp>(*op)) {
            return vectorizeBinOp(ewSubOp);
        } else if (auto ewDivOp = dyn_cast<daphne::EwDivOp>(*op)) {
            auto lhs = buildVectorized(ewDivOp.getOperand(0).getDefiningOp());
            auto rhs = buildVectorized(ewDivOp.getOperand(1).getDefiningOp());

            if (lhs.has_value() && rhs.has_value()) {
                // needs typechecking (lhs must be float)
                if (lhs->getType().isa<FloatType>()) {
                    auto built_val =
                        builder.create<daphne::EwDivOp>(loc, *lhs, *rhs)
                            .getResult();
                    builtVals.push_back(built_val);
                    return built_val;
                }
            }

            return std::nullopt;
        } else if (auto ewPowOp = dyn_cast<daphne::EwPowOp>(*op)) {
            return vectorizeBinOpLmatrix(ewPowOp);
        } else if (auto ewModOp = dyn_cast<daphne::EwModOp>(*op)) {
            return vectorizeBinOpLmatrix(ewModOp);
        } else if (auto ewAndOp = dyn_cast<daphne::EwAndOp>(*op)) {
            return vectorizeBinOpLmatrix(ewAndOp);
        } else if (auto ewOrOp = dyn_cast<daphne::EwOrOp>(*op)) {
            return vectorizeBinOpLmatrix(ewOrOp);
        } else if (auto ewLogOp = dyn_cast<daphne::EwLogOp>(*op)) {
            return vectorizeBinOpLmatrix(ewLogOp);
        } else if (auto ewMinOp = dyn_cast<daphne::EwMinOp>(*op)) {
            return vectorizeBinOpLmatrix(ewMinOp);
        } else if (auto ewMaxOp = dyn_cast<daphne::EwMaxOp>(*op)) {
            return vectorizeBinOpLmatrix(ewMaxOp);
        } else if (auto ewLeOp = dyn_cast<daphne::EwLeOp>(*op)) {
            return vectorizeBinOpLmatrix(ewLeOp);
        } else if (auto ewLtOp = dyn_cast<daphne::EwLtOp>(*op)) {
            return vectorizeBinOpLmatrix(ewLtOp);
        } else if (auto ewGeOp = dyn_cast<daphne::EwGeOp>(*op)) {
            return vectorizeBinOpLmatrix(ewGeOp);
        } else if (auto ewGtOp = dyn_cast<daphne::EwGtOp>(*op)) {
            return vectorizeBinOpLmatrix(ewGtOp);
        } else if (auto ewEqOp = dyn_cast<daphne::EwEqOp>(*op)) {
            return vectorizeBinOpLmatrix(ewEqOp);
        } else if (auto ewNeqOp = dyn_cast<daphne::EwNeqOp>(*op)) {
            return vectorizeBinOpLmatrix(ewNeqOp);
        } else if (auto ewAbsOp = dyn_cast<daphne::EwAbsOp>(*op)) {
            return vectorizeUnOp(ewAbsOp);
        } else if (auto ewSignOp = dyn_cast<daphne::EwSignOp>(*op)) {
            return vectorizeUnOp(ewSignOp);
        } else if (auto ewExpOp = dyn_cast<daphne::EwExpOp>(*op)) {
            return vectorizeUnOp(ewExpOp);
        } else if (auto ewLnOp = dyn_cast<daphne::EwLnOp>(*op)) {
            return vectorizeUnOp(ewLnOp);
        } else if (auto ewSqrtOp = dyn_cast<daphne::EwSqrtOp>(*op)) {
            return vectorizeUnOp(ewSqrtOp);
        } else if (auto ewSinOp = dyn_cast<daphne::EwSinOp>(*op)) {
            return vectorizeUnOp(ewSinOp);
        } else if (auto ewCosOp = dyn_cast<daphne::EwCosOp>(*op)) {
            return vectorizeUnOp(ewCosOp);
        } else if (auto ewTanOp = dyn_cast<daphne::EwTanOp>(*op)) {
            return vectorizeUnOp(ewTanOp);
        } else if (auto ewAsinOp = dyn_cast<daphne::EwAsinOp>(*op)) {
            return vectorizeUnOp(ewAsinOp);
        } else if (auto ewAcosOp = dyn_cast<daphne::EwAcosOp>(*op)) {
            return vectorizeUnOp(ewAcosOp);
        } else if (auto ewAtanOp = dyn_cast<daphne::EwAtanOp>(*op)) {
            return vectorizeUnOp(ewAtanOp);
        } else if (auto ewSinhOp = dyn_cast<daphne::EwSinhOp>(*op)) {
            return vectorizeUnOp(ewSinhOp);
        } else if (auto ewCoshOp = dyn_cast<daphne::EwCoshOp>(*op)) {
            return vectorizeUnOp(ewCoshOp);
        } else if (auto ewTanhOp = dyn_cast<daphne::EwTanhOp>(*op)) {
            return vectorizeUnOp(ewTanhOp);
        }

        return std::nullopt;
    }

  private:
    InnerLoop(scf::ForOp &loop,                         // The loop itself
              const long int rows, const long int cols, // Matrix dimensions
              mlir::Value i, // Induction variable of the outer loop
              mlir::Value j, mlir::Location loc,
              OpBuilder &builder) // Module and builder
        : scf::ForOp(loop), rows(rows), cols(cols), i(i), j(j), loc(loc),
          builder(builder) {}

    // Check if a SliceColOp accesses the i, j matrix element and return the
    // original matrix if so
    std::optional<mlir::Value> ijAccess(daphne::SliceColOp &sliceColOp) {
        auto sliceRowOp = static_cast<daphne::SliceRowOp>(
            sliceColOp.getSource().getDefiningOp());
        if (auto sourceType = dyn_cast<daphne::MatrixType>(
                sliceRowOp.getSource().getType())) {
            if (sourceType.getNumRows() != rows ||
                sourceType.getNumCols() != cols) {
                return std::nullopt;
            }
        }

        auto source = sliceRowOp.getSource();
        if (auto blockArgument = dyn_cast<mlir::BlockArgument>(source)) {
            // If this is an iteration argument, we need to get its
            // original source from outside the outer loop
            if (blockArgument.getOwner() == getBody()) {
                auto outerLoop = static_cast<scf::ForOp>(
                    blockArgument.getOwner()
                        ->getParentOp()
                        ->getParentOp()); // safe because we know we're in a
                                          // nested loop
                source =
                    outerLoop.getInitArgs()[blockArgument.getArgNumber() -
                                            1]; // - 1 because the InductionVar
                                                // is counted in the ArgNumber
            }
        }

        if (isi(sliceRowOp.getOperand(1)) &&
            isiplus1(sliceRowOp.getOperand(2)) &&
            isj(sliceColOp.getOperand(1)) &&
            isjplus1(sliceColOp.getOperand(2))) {
            return source;
        }

        return std::nullopt;
    }

    // Helper functions
    bool isi(mlir::Value val) { return i == resolveIndex(val); }
    bool isj(mlir::Value val) { return j == resolveIndex(val); }

    bool isiplus1(mlir::Value val) {
        if (auto castOp = dyn_cast<daphne::CastOp>(val.getDefiningOp())) {
            if (auto addOp = dyn_cast<daphne::EwAddOp>(
                    castOp.getOperand().getDefiningOp())) {
                if (isi(addOp.getOperand(0)) &&
                    isConstantInteger(addOp.getOperand(1), 1)) {
                    return true;
                }
            }
        }

        return false;
    }

    bool isjplus1(mlir::Value val) {
        if (auto castOp = dyn_cast<daphne::CastOp>(val.getDefiningOp())) {
            if (auto addOp = dyn_cast<daphne::EwAddOp>(
                    castOp.getOperand().getDefiningOp())) {
                if (isj(addOp.getOperand(0)) &&
                    isConstantInteger(addOp.getOperand(1), 1)) {
                    return true;
                }
            }
        }

        return false;
    }

    // Vectorize a binary operation
    template <typename T> std::optional<mlir::Value> vectorizeBinOp(T binOp) {
        auto lhs = buildVectorized(binOp.getOperand(0).getDefiningOp());
        auto rhs = buildVectorized(binOp.getOperand(1).getDefiningOp());

        if (lhs.has_value() && rhs.has_value()) {
            auto built_val = builder.create<T>(loc, *lhs, *rhs).getResult();
            builtVals.push_back(built_val);
            return built_val;
        }
        return std::nullopt;
    }

    // Vectorize a binary operation, ensuring that the lhs is a matrix
    template <typename T>
    std::optional<mlir::Value> vectorizeBinOpLmatrix(T binOp) {
        auto lhs = buildVectorized(binOp.getOperand(0).getDefiningOp());
        auto rhs = buildVectorized(binOp.getOperand(1).getDefiningOp());

        if (lhs.has_value() && rhs.has_value()) {
            // Some operations are not defined when lhs is a scalar, so we check
            // for this
            if (dyn_cast<daphne::MatrixType>(lhs->getType())) {
                auto built_val = builder.create<T>(loc, *lhs, *rhs).getResult();
                builtVals.push_back(built_val);
                return built_val;
            }
        }

        return std::nullopt;
    }

    // Vectorize a unary operation
    template <typename T> std::optional<mlir::Value> vectorizeUnOp(T unOp) {
        auto elementType = unOp.getResult().getType();
        if (auto matrixType = dyn_cast<daphne::MatrixType>(elementType)) {
            elementType = matrixType.getElementType();
        }

        if (auto operand = buildVectorized(unOp.getOperand().getDefiningOp())) {
            // Make sure to preserve the matrix element type
            auto type = operand->getType();
            if (auto matrixType = dyn_cast<daphne::MatrixType>(type)) {
                type = matrixType.withElementType(elementType);
            }
            auto built_val = builder.create<T>(loc, type, *operand).getResult();
            builtVals.push_back(built_val);
            return built_val;
        }
        return std::nullopt;
    }

    std::vector<mlir::Value> builtVals;

    const long int rows;
    const long int cols;

    mlir::Value i;
    mlir::Value j;

    mlir::Location loc;
    mlir::OpBuilder builder;
};

void LoopVectorizationPass::runOnOperation() {
    auto module = getOperation();
    auto loc = module.getLoc();
    mlir::OpBuilder builder(module);

    module.walk([&](mlir::Operation *op) {
        if (auto loopOp = dyn_cast<scf::ForOp>(op)) {
            // Set insertion point right before the loop
            builder.setInsertionPoint(loopOp);

            // Check if inner loop could potentially be vectorized
            if (auto innerLoop =
                    InnerLoop::getCompatibleInnerLoop(loopOp, loc, builder)) {
                auto yieldOp = static_cast<scf::YieldOp>(
                    &innerLoop->getBody()
                         ->getOperations()
                         .back()); // safe because we know the last operation is
                                   // a YieldOp

                // New results for the YieldOp
                std::vector<mlir::Value> newResults;

                for (auto yieldRes : yieldOp.getOperands()) {
                    if (auto insertedVal =
                            innerLoop->ijInsert(yieldRes.getDefiningOp())) {
                        if (auto vectorized = innerLoop->buildVectorized(
                                insertedVal->getDefiningOp())) {

                            // If type conflicts happen, don't vectorize
                            if (yieldRes.getType() == vectorized->getType()) {
                                newResults.push_back(*vectorized);
                            }
                        }
                    }
                }
                // Only commit results if we successfully vectorized all of them
                if (newResults.size() == yieldOp.getNumOperands()) {
                    for (uint i = 0; i < newResults.size(); i++) {
                        auto result = newResults[i];
                        // Replace the old results with the new ones (even
                        // though we delete the yield, this is useful for loops
                        // that return multiple interdependent values)
                        yieldOp.getOperand(i).replaceAllUsesWith(result);
                        // Replace the old results with the new ones outside of
                        // the loop
                        loopOp.getResult(i).replaceAllUsesWith(result);
                    }
                    // Delete the unnecessary casts
                    innerLoop->flushBuiltCasts();
                    // Delete the entire loop
                    loopOp.erase();
                } else {
                    // Failed to vectorize yields, delete all the extra values
                    // we created
                    innerLoop->flushBuiltVals();
                }
            }
        }
    });
}

std::unique_ptr<Pass> daphne::createLoopVectorizationPass() {
    return std::make_unique<LoopVectorizationPass>();
}
