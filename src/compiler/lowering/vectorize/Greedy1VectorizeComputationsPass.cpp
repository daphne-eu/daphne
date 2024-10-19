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

#include "api/cli/DaphneUserConfig.h"
#include "compiler/lowering/vectorize/VectorUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneVectorizableOpInterface.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cstdint>
#include <mlir/IR/OpDefinition.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/ADT/STLExtras.h>

#include <spdlog/spdlog.h>
#include <util/ErrorHandler.h>

using namespace mlir;

namespace {

//-----------------------------------------------------------------
// CONST
//-----------------------------------------------------------------


//-----------------------------------------------------------------
// Class functions
//-----------------------------------------------------------------

struct Greedy1VectorizeComputationsPass
    : public PassWrapper<Greedy1VectorizeComputationsPass, OperationPass<func::FuncOp>> {
    void runOnOperation() final;

    const DaphneUserConfig& userConfig;

    explicit Greedy1VectorizeComputationsPass(const DaphneUserConfig& cfg) : userConfig(cfg) {}
};

void printStack(std::stack<std::tuple<mlir::Operation *, Pipeline *, Pipeline *>> s) {
    llvm::outs() << "[";
    while (!s.empty()) {
        auto op = s.top();
        llvm::outs() << "(" << std::get<0>(op)->getName().getStringRef().str() << ", " << std::get<1>(op) << "), ";
        s.pop();
    }
    llvm::outs() << "]\n";
}

void printGraph(std::vector<mlir::Operation *> leafOps, std::string filename) {
    std::stack<mlir::Operation *> stack;
    std::ofstream dot(filename);
    if (!dot.is_open()) {
        throw std::runtime_error("test");
    }

    dot << "digraph G {\n";
    for (auto leaf : leafOps) {
        stack.push(leaf);
    }

    std::vector<mlir::Operation *> visited;

    while (!stack.empty()) {
        auto op = stack.top();
        stack.pop();
        if (std::find(visited.begin(), visited.end(), op) != visited.end()) {
            continue;
        }
        visited.push_back(op);

        auto v = llvm::dyn_cast<daphne::Vectorizable>(op);
        for (unsigned i = 0; i < v->getNumOperands(); ++i) {
            mlir::Value e = v->getOperand(i);
            auto defOp = e.getDefiningOp();
            if (llvm::isa<daphne::MatrixType>(e.getType()) && llvm::isa<daphne::Vectorizable>(defOp)) {
                dot << "\"" << defOp->getName().getStringRef().str() << "+" << std::hex
                    << reinterpret_cast<uintptr_t>(defOp) << "\" -> \"" << op->getName().getStringRef().str() << "+"
                    << std::hex << reinterpret_cast<uintptr_t>(op) << "\" [label=\"" << i << "\"];\n";
                stack.push(defOp);
            }
        }
    }
    dot << "}";
    dot.close();
}
} // namespace

void Greedy1VectorizeComputationsPass::runOnOperation() {

    auto func = getOperation();

    VectorIndex ZeroDecision = 0;
    /*if (userConfig.colFirst) {
        ZeroDecision = 1;
    }*/

    std::vector<mlir::Operation *> ops;
    func->walk([&](daphne::Vectorizable op) {
        for (auto opType : op->getOperandTypes()) {
            if (!opType.isIntOrIndexOrFloat() && !llvm::isa<daphne::StringType>(opType)) {
                ops.emplace_back(op);
                break;
            }
        }
    });
    std::reverse(ops.begin(), ops.end());

    // result
    std::vector<Pipeline *> pipelines;
    std::vector<mlir::Operation *> leafOps;
    std::stack<std::tuple<mlir::Operation *, Pipeline *, DisconnectReason>> stack;

    for (const auto &op : ops) {
        auto users = op->getUsers();
        bool found = false;
        for (auto u : users) {
            if (std::find(ops.begin(), ops.end(), u) != ops.end()) {
                found = true;
                break;
            }
        }
        if (!found) {
            leafOps.push_back(op);
            stack.push({op, nullptr, DisconnectReason::INVALID});
        }
    }

    std::multimap<PipelinePair, DisconnectReason> mmProducerConsumerRelationships;
    std::map<mlir::Operation *, Pipeline *> operationToPipeline;

    // std::vector<mlir::Operation*> boundingOperations;

    while (!stack.empty()) {
        auto t = stack.top();
        stack.pop();
        auto op = std::get<0>(t);
        auto currPipeline = std::get<1>(t);
        auto disReason = std::get<2>(t);

        if (operationToPipeline.find(op) != operationToPipeline.end()) {
            auto producerPipeline = operationToPipeline.at(op);
            mmProducerConsumerRelationships.insert({{currPipeline, producerPipeline}, disReason});
            continue;
        }

        if (disReason != DisconnectReason::NONE) {
            auto _pipeline = new Pipeline();
            pipelines.push_back(_pipeline);

            // check needed for empty init
            if (currPipeline != nullptr)
                mmProducerConsumerRelationships.insert({{currPipeline, _pipeline}, disReason});

            currPipeline = _pipeline;
        }

        operationToPipeline.insert({op, currPipeline});
        currPipeline->push_back(op);

        auto vectOp = llvm::dyn_cast<daphne::Vectorizable>(op);

        for (size_t i = 0; i < vectOp->getNumOperands(); ++i) {
            auto operand = vectOp->getOperand(i);

            // llvm::outs() << op->getName().getStringRef().str() << " ";

            if (!llvm::isa<daphne::MatrixType>(operand.getType()))
                continue;

            if (llvm::isa<mlir::BlockArgument>(operand)) {
                continue;
            }

            // could it help to check if we check if operand.getDefiningOp is inside (global) ops vector?
            if (auto vectDefOp = llvm::dyn_cast<daphne::Vectorizable>(operand.getDefiningOp())) {
                // llvm::outs() << vectDefOp->getName().getStringRef().str() << "\n";

                auto split = vectOp.getVectorSplits()[ZeroDecision][i];
                auto combine = vectDefOp.getVectorCombines()[ZeroDecision][0];

                // same block missing
                if (VectorUtils::matchingVectorSplitCombine(split, combine) &&
                    vectDefOp->getBlock() == vectOp->getBlock()) {
                    if (vectDefOp->hasOneUse()) {
                        stack.push({vectDefOp, currPipeline, DisconnectReason::NONE});
                    } else {
                        stack.push({vectDefOp, currPipeline, DisconnectReason::MULTIPLE_CONSUMERS});
                    }
                } else {
                    stack.push({vectDefOp, currPipeline, DisconnectReason::INVALID});
                }
            } else {
                // defOp is outside of consideration, top horz. fusion possible
                // boundingOperations.push_back(op);
                // llvm::outs() << "\n";
            }
        }
    }

    // Needed as Greedy1 is only considering the first possiblity
    std::map<mlir::Operation *, size_t> decisionIxs;
    for (const auto &op : ops) {
        decisionIxs.insert({op, ZeroDecision});
    }

    // mmPCR to PCR
    std::map<PipelinePair, DisconnectReason> producerConsumerRelationships =
        VectorUtils::consolidateProducerConsumerRelationship(mmProducerConsumerRelationships);

    VectorUtils::greedyMergePipelinesProducerConsumer(pipelines, operationToPipeline, producerConsumerRelationships);

    // VectorUtils::DEBUG::printPipelines(pipelines);

    // Post Processing

    std::vector<Pipeline> _pipelines;
    _pipelines.resize(pipelines.size());

    std::transform(pipelines.begin(), pipelines.end(), _pipelines.begin(), [](const auto &ptr) { return *ptr; });

    // will crash if for some reason the pipelines itself are not topologically sorted
    VectorUtils::createVectorizedPipelineOps(func, _pipelines, decisionIxs);

    return;

}

std::unique_ptr<Pass> daphne::createGreedy1VectorizeComputationsPass(const DaphneUserConfig& cfg) {
    return std::make_unique<Greedy1VectorizeComputationsPass>(cfg);
}