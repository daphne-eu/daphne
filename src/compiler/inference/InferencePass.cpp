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

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

#include <stdexcept>
#include <memory>
#include <vector>
#include <utility>
#include <iostream>

using namespace mlir;

daphne::InferenceConfig::InferenceConfig(bool partialInferenceAllowed,
                                         bool typeInference,
                                         bool shapeInference,
                                         bool frameLabelInference,
                                         bool sparsityInference)
    : partialInferenceAllowed(partialInferenceAllowed), typeInference(typeInference), shapeInference(shapeInference),
      frameLabelInference(frameLabelInference), sparsityInference(sparsityInference) {}

namespace {
    /**
     * @brief Removes properties that are fragile to changes in SCF operations (currently only sparsity). This
     * ensures that they are valid in loop bodies. This is done inserting a `CastOp`
     * @param opOperand The operand to the SCF operation
     * @return the new type without the properties
     */
    Type removeSCFVariantProperties(OpOperand &opOperand) {
        auto ty = opOperand.get().getType();
        auto matTy = ty.dyn_cast<daphne::MatrixType>();
        if(matTy && matTy.getSparsity() != -1.0) {
            OpBuilder builder(opOperand.getOwner());
            auto castOp = builder
                .create<daphne::CastOp>(opOperand.getOwner()->getLoc(), matTy.withSparsity(-1.0), opOperand.get());
            opOperand.set(castOp);
            return castOp.getType();
        }
        return ty;
    }
}

/**
 * @brief A compiler pass infering various properties of the data objects.
 * 
 * Rooted at a function, the pass walks the operations, and for each operation
 * it encounters, it infers all currently considered properties of the
 * operation's results based on the properties of the operation's arguments.
 * This approach can easily handle dependencies between different properties to
 * be infered without explicitly modeling them.
 * 
 * Note that the actual inference logic is outsourced to MLIR operation
 * interfaces.
 */
// TODO Currently, the properties to be inferred are hardcoded, but we should
// make them configurable, whereby different instances of this pass should be
// able to infer different sets of properties.
class InferencePass : public PassWrapper<InferencePass, FunctionPass> {
    daphne::InferenceConfig cfg;

    std::function<WalkResult(Operation*)> walkOp = [&](Operation * op) {
        // Type inference.
        if(returnsUnknownType(op))
            daphne::setInferedTypes(op, cfg.partialInferenceAllowed);

        // Frame label inference.
        if(cfg.frameLabelInference && returnsFrameWithUnknownLabels(op)) {
            if(auto inferFrameLabelsOp = llvm::dyn_cast<daphne::InferFrameLabels>(op))
                inferFrameLabelsOp.inferFrameLabels();
            // Else: Not a problem, since currently we use the frame labels
            // only to aid type inference, and for this purpose, we don't
            // need the labels in all cases.
        }

        // Shape or Sparsity inference.
        bool doShapeInference = cfg.shapeInference && returnsUnknownShape(op);
        bool doSparsityInference = cfg.sparsityInference && returnsUnknownSparsity(op);
        if(doShapeInference || doSparsityInference) {
            const bool isScfOp = op->getDialect() == op->getContext()->getOrLoadDialect<scf::SCFDialect>();
            // ----------------------------------------------------------------
            // Handle all non-SCF operations
            // ----------------------------------------------------------------
            if(!isScfOp) {
                if (doShapeInference) {
                    // Try to infer the shapes of all results of this operation.
                    std::vector<std::pair<ssize_t, ssize_t>> shapes = daphne::tryInferShape(op);
                    const size_t numRes = op->getNumResults();
                    if(shapes.size() != numRes)
                        throw std::runtime_error(
                            "shape inference for op " +
                                op->getName().getStringRef().str() + " returned " +
                                std::to_string(shapes.size()) + " shapes, but the "
                                                                "op has " + std::to_string(numRes) + " results"
                        );
                    // Set the infered shapes on all results of this operation.
                    for(size_t i = 0 ; i < numRes ; i++) {
                        const ssize_t numRows = shapes[i].first;
                        const ssize_t numCols = shapes[i].second;
                        Value rv = op->getResult(i);
                        const Type rt = rv.getType();
                        if(auto mt = rt.dyn_cast<daphne::MatrixType>())
                            rv.setType(mt.withShape(numRows, numCols));
                        else if(auto ft = rt.dyn_cast<daphne::FrameType>())
                            rv.setType(ft.withShape(numRows, numCols));
                        else
                            throw std::runtime_error(
                                "shape inference cannot set the shape of op " +
                                    op->getName().getStringRef().str() +
                                    " operand " + std::to_string(i) + ", since it "
                                                                      "is neither a matrix nor a frame"
                            );
                    }
                }
                if (doSparsityInference) {
                    // Try to infer the sparsity of all results of this operation.
                    std::vector<double> sparsities = daphne::tryInferSparsity(op);
                    const size_t numRes = op->getNumResults();
                    if(sparsities.size() != numRes)
                        throw std::runtime_error(
                            "sparsity inference for op " +
                                op->getName().getStringRef().str() + " returned " +
                                std::to_string(sparsities.size()) + " shapes, but the "
                                                                "op has " + std::to_string(numRes) + " results"
                        );
                    // Set the inferred sparsities on all results of this operation.
                    for(size_t i = 0 ; i < numRes ; i++) {
                        const double sparsity = sparsities[i];
                        Value rv = op->getResult(i);
                        const Type rt = rv.getType();
                        auto mt = rt.dyn_cast<daphne::MatrixType>();
                        auto ft = rt.dyn_cast<daphne::FrameType>();
                        if(mt)
                            rv.setType(mt.withSparsity(sparsity));
                        else if((ft && sparsity != -1) || !ft)
                            // We do not support sparsity for frames, but if the
                            // sparsity for a frame result is provided as
                            // unknown (-1) that's okay.
                            throw std::runtime_error(
                                "sparsity inference cannot set the shape of op " +
                                    op->getName().getStringRef().str() +
                                    " operand " + std::to_string(i) + ", since it "
                                                                      "is not a matrix"
                            );
                    }
                }
            }
            // ----------------------------------------------------------------
            // Special treatment for some SCF operations
            // ----------------------------------------------------------------
            // TODO In the future, we should support changing the shape of some
            // data object within a loop's body as well as mismatching shapes
            // after then/else branches.
            else if(auto whileOp = llvm::dyn_cast<scf::WhileOp>(op)) {
                Block & beforeBlock = whileOp.before().front();
                Block & afterBlock = whileOp.after().front();
                // Transfer the WhileOp's operand types to the block arguments
                // and results to fulfill constraints on the WhileOp.
                for(size_t i = 0; i < whileOp.getNumOperands(); i++) {
                    Type t = removeSCFVariantProperties(whileOp->getOpOperand(i));
                    beforeBlock.getArgument(i).setType(t);
                    afterBlock.getArgument(i).setType(t);
                    whileOp.getResult(i).setType(t);
                }
                // Continue the walk on both blocks of the WhileOp. We trigger
                // this explicitly, since we need to do something afterwards.
                beforeBlock.walk<WalkOrder::PreOrder>(walkOp);
                afterBlock.walk<WalkOrder::PreOrder>(walkOp);
                // Check if the infered types match the required result types.
                // This is not the case if, for instance, the shape of some
                // variable written in the loop changes. The WhileOp would also
                // check this later during verification, but here, we want to
                // throw a readable error message.
                Operation * yieldOp = afterBlock.getTerminator();
                for(size_t i = 0; i < whileOp.getNumOperands(); i++) {
                    Type yieldedTy = removeSCFVariantProperties(yieldOp->getOpOperand(i));
                    Type resultTy = op->getResult(i).getType();
                    if(yieldedTy != resultTy)
                        throw std::runtime_error(
                                "the type/shape of a variable must not be "
                                "changed within the body of a while-loop"
                        );
                }
                // Tell the walker to skip the descendants of the WhileOp, we
                // have already triggered a walk on them explicitly.
                return WalkResult::skip();
            }
            else if(auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
                Block & block = forOp.region().front();
                const size_t numIndVars = forOp.getNumInductionVars();
                // Transfer the ForOp's operand types to the block arguments
                // and results to fulfill constraints on the ForOp.
                for(size_t i = 0; i < forOp.getNumIterOperands(); i++) {
                    Type t = removeSCFVariantProperties(forOp.getIterOpOperands()[i]);
                    block.getArgument(i + numIndVars).setType(t);
                    forOp.getResult(i).setType(t);
                }
                // Continue the walk on the body block of the ForOp. We trigger
                // this explicitly, since we need to do something afterwards.
                block.walk<WalkOrder::PreOrder>(walkOp);
                // Check if the infered types match the required result types.
                // This is not the case if, for instance, the shape of some
                // variable written in the loop changes. The ForOp would also
                // check this later during verification, but here, we want to
                // throw a readable error message.
                Operation * yieldOp = block.getTerminator();
                for(size_t i = 0; i < forOp.getNumIterOperands(); i++) {
                    Type yieldedTy = removeSCFVariantProperties(yieldOp->getOpOperand(i));
                    Type resultTy = op->getResult(i).getType();
                    if(yieldedTy != resultTy)
                        throw std::runtime_error(
                                "the type/shape of a variable must not be "
                                "changed within the body of a for-loop"
                        );
                }
                // Tell the walker to skip the descendants of the ForOp, we
                // have already triggered a walk on them explicitly.
                return WalkResult::skip();
            }
            else if(auto ifOp = llvm::dyn_cast<scf::IfOp>(op)) {
                // Walk the then/else blocks first. We need the inference on
                // them before we can do anything about the IfOp itself.
                ifOp.thenBlock()->walk<WalkOrder::PreOrder>(walkOp);
                ifOp.elseBlock()->walk<WalkOrder::PreOrder>(walkOp);
                // Check if the yielded types/shapes are the same in both
                // branches. The IfOp would also check this later during
                // verification, but here, we want to throw a readable error
                // message.
                // Additionally, we set the result types of the IfOp here.
                scf::YieldOp thenYield = ifOp.thenYield();
                scf::YieldOp elseYield = ifOp.elseYield();
                for(size_t i = 0; i < ifOp.getNumResults(); i++) {
                    Type thenTy = removeSCFVariantProperties(thenYield->getOpOperand(i));
                    Type elseTy = removeSCFVariantProperties(elseYield->getOpOperand(i));
                    if(thenTy != elseTy)
                        throw std::runtime_error(
                                "a variable must not be assigned values of "
                                "different types/shapes in then/else branches"
                        );
                    ifOp.getResult(i).setType(thenTy);
                }
                // Tell the walker to skip the descendants of the IfOp, we
                // have already triggered a walk on them explicitly.
                return WalkResult::skip();
            }
        }
        // Continue the walk normally.
        return WalkResult::advance();
    };

public:
    InferencePass(daphne::InferenceConfig cfg) : cfg(cfg) {}

    void runOnFunction() override {
        getFunction().walk<WalkOrder::PreOrder>(walkOp);
        // infer function return types
        getFunction().setType(FunctionType::get(&getContext(),
            getFunction().getType().getInputs(),
            getFunction().body().back().getTerminator()->getOperandTypes()));
    }

    static bool returnsUnknownType(Operation *op) {
        return llvm::any_of(op->getResultTypes(), [](Type resType) {
            if(resType.isa<daphne::UnknownType>())
                return true;
            if(auto mt = resType.dyn_cast<daphne::MatrixType>())
                return mt.getElementType().isa<daphne::UnknownType>();
            if(auto ft = resType.dyn_cast<daphne::FrameType>())
                for(Type ct : ft.getColumnTypes())
                    if(ct.isa<daphne::UnknownType>())
                        return true;
            return false;
        });
    }

    static bool returnsFrameWithUnknownLabels(Operation * op) {
        return llvm::any_of(op->getResultTypes(), [](Type resultType) {
            auto ft = resultType.dyn_cast<daphne::FrameType>();
            return ft && !ft.getLabels();
        });
    }

    static bool returnsUnknownShape(Operation * op) {
        return llvm::any_of(op->getResultTypes(), [](Type rt) {
            if(auto mt = rt.dyn_cast<daphne::MatrixType>())
                return mt.getNumRows() == -1 || mt.getNumCols() == -1;
            if(auto ft = rt.dyn_cast<daphne::FrameType>())
                return ft.getNumRows() == -1 || ft.getNumCols() == -1;
            return false;
        });
    }

    static bool returnsUnknownSparsity(Operation * op) {
        return llvm::any_of(op->getResultTypes(), [](Type rt) {
            if(auto mt = rt.dyn_cast<daphne::MatrixType>())
                return mt.getSparsity() == -1.0;
            return false;
        });
    }
};

std::unique_ptr<Pass> daphne::createInferencePass(daphne::InferenceConfig cfg) {
    return std::make_unique<InferencePass>(cfg);
}