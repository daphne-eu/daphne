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

#pragma once
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneVectorizableOpInterface.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <llvm/ADT/SmallVector.h>
#include <map>
#include <queue>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <thread>

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/IR/PassManagerInternal.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ManagedStatic.h"

using VectorIndex = std::size_t;
using Pipeline = std::vector<mlir::Operation *>;
using PipelinePair = std::pair<Pipeline *, Pipeline *>;

using PipelineOpPair = std::pair<mlir::daphne::VectorizedPipelineOp, mlir::daphne::VectorizedPipelineOp>;

namespace std {
template <> struct hash<PipelinePair> {
    size_t operator()(const PipelinePair &p) const {
        return std::hash<Pipeline *>{}(p.first) ^ std::hash<Pipeline *>{}(p.second);
    }
};
} // namespace std

enum class DisconnectReason { NONE, MULTIPLE_CONSUMERS, INVALID };

enum class EdgeStatus { INVALID, ACTIVE, INACTIVE };

struct VectorUtils {

    /**
     * @brief Checks if a VectorSplit and a VectorCombine are compatible.
     *
     * This function compares the provided VectorSplit and VectorCombine to
     * determine if they match by remapping the split to a matching combine.
     * Compatible pairs are ROWS-ROWS and COLS-COLS.
     *
     * @param split VectorSplit value representing the split of a operation.
     * @param combine VectorCombine value representing the combine of a operation.
     * @return true, if VectorSplit and VectorCombine are compabitlbe
     * @return false, otherwise
     */

    static bool matchingVectorSplitCombine(mlir::daphne::VectorSplit split, mlir::daphne::VectorCombine combine) {
        // llvm::outs() << split << " " << combine << " ";
        mlir::daphne::VectorCombine _operandCombine;
        switch (split) {
        case mlir::daphne::VectorSplit::ROWS:
            _operandCombine = mlir::daphne::VectorCombine::ROWS;
            break;
        case mlir::daphne::VectorSplit::COLS:
            _operandCombine = mlir::daphne::VectorCombine::COLS;
            break;
        default:
            // No matching split/combine; basically resulting in separate pipelines
            return false;
        }
        if (combine == _operandCombine) {
            return true;
        }
        return false;
    }

    // Greedy merge along (valid) MULTIPLE_CONSUMER relationships
    // by checking if resulting pipelines can be sorted topologically.
    static void
    greedyMergePipelinesProducerConsumer(std::vector<Pipeline *> &pipelines,
                                         std::map<mlir::Operation *, Pipeline *> &operationToPipeline,
                                         std::map<PipelinePair, DisconnectReason> &producerConsumerRelationships) {
        bool change = true;
        while (change) {
            change = false;

            std::multimap<PipelinePair, DisconnectReason> mmPCR;
            for (const auto &[pipePair, disReason] : producerConsumerRelationships) {

                if (disReason == DisconnectReason::INVALID)
                    continue;

                if (VectorUtils::tryTopologicalSortMerged(pipelines, producerConsumerRelationships, pipePair.first,
                                                          pipePair.second)) {
                    auto mergedPipeline =
                        VectorUtils::mergePipelines(pipelines, operationToPipeline, pipePair.first, pipePair.second);

                    for (const auto &[_pipePair, _disReason] : producerConsumerRelationships) {

                        // Ignore in case that is current pair is pipePair
                        if (_pipePair.first == pipePair.first && _pipePair.second == pipePair.second)
                            continue;

                        // Rewrite Relationships
                        if (_pipePair.first == pipePair.first || _pipePair.first == pipePair.second) {
                            auto newPipePair = std::make_pair(mergedPipeline, _pipePair.second);
                            mmPCR.insert({newPipePair, _disReason});
                        } else if (_pipePair.second == pipePair.first || _pipePair.second == pipePair.second) {
                            auto newPipePair = std::make_pair(_pipePair.first, mergedPipeline);
                            mmPCR.insert({newPipePair, _disReason});
                        } else {
                            mmPCR.insert({_pipePair, _disReason});
                        }
                    }

                    change = true;
                    break;
                }
            }

            // In case of no change the mmPCR is not filled, ignore
            if (change)
                producerConsumerRelationships = VectorUtils::consolidateProducerConsumerRelationship(mmPCR);

            // VectorUtils::DEBUG::printPCR(producerConsumerRelationships);
            // VectorUtils::DEBUG::printPipelines(pipelines);
        }
    }

    //------------------------------------------------------------------------------

    // Function merges two pipelines into one by appending all operations from one pipeline to another
    // Order is not really considered, as it is embodied in IR
    static void mergePipelines(std::vector<Pipeline *> &pipelines,
                               std::map<mlir::Operation *, size_t> &operationToPipelineIx, size_t pipeIx1,
                               size_t pipeIx2) {
        // llvm::outs() << mergeFromIx << " " << mergeIntoIx << "\n";
        if (pipeIx1 == pipeIx2)
            return;
        if (pipeIx2 > pipeIx1)
            std::swap(pipeIx1, pipeIx2);

        std::vector<mlir::Operation *> *mergedPipeline(pipelines.at(pipeIx2));
        for (auto op : *pipelines.at(pipeIx1)) {
            if (std::find(mergedPipeline->begin(), mergedPipeline->end(), op) == mergedPipeline->end()) {
                mergedPipeline->push_back(op);
                operationToPipelineIx[op] = pipeIx2;
            }
        }
        pipelines.at(pipeIx2) = std::move(mergedPipeline);
        pipelines.erase(pipelines.begin() + pipeIx1);
    }

    static Pipeline *mergePipelines(std::vector<Pipeline *> &pipelines,
                                    std::map<mlir::Operation *, Pipeline *> &operationToPipeline, Pipeline *pipe1,
                                    Pipeline *pipe2) {
        if (pipe1 == pipe2)
            return nullptr;

        for (auto op : *pipe2) {
            if (std::find(pipe1->begin(), pipe1->end(), op) == pipe1->end()) {
                pipe1->push_back(op);
                operationToPipeline[op] = pipe1;
            }
        }

        auto pipeIx2 = std::find(pipelines.begin(), pipelines.end(), pipe2);
        pipelines.erase(pipeIx2);
        return pipe1;
    }

    // only works if pipeline ops are topologically sorted in reverse
    static bool arePipelineOpsDependent(mlir::daphne::VectorizedPipelineOp pipeOp1,
                                        mlir::daphne::VectorizedPipelineOp pipeOp2) {

        if (pipeOp1 == pipeOp2)
            return true;

        std::stack<mlir::Operation *> s;
        std::unordered_set<mlir::Operation *> visited;

        s.push(pipeOp1);
        while (!s.empty()) {
            mlir::Operation *currOp = s.top();
            s.pop();

            // Connection found
            if (currOp == pipeOp2)
                return true;

            if (visited.insert(currOp).second) {
                for (const auto &operand : currOp->getOperands()) {
                    if (auto defOp = operand.getDefiningOp()) {
                        s.push(defOp);
                    }
                }
            }
        }

        return false;
    }

    static bool tryTopologicalSortMerged(std::vector<Pipeline *> &pipelines,
                                         std::map<PipelinePair, DisconnectReason> &rel, Pipeline *pipe1,
                                         Pipeline *pipe2) {

        // if (pipe2 > pipe1)
        //   std::swap(pipe1, pipe2);

        // prealloc
        std::map<Pipeline *, std::unordered_set<Pipeline *>> pipeline_graph;
        for (auto pipe : pipelines) {
            if (pipe == pipe1)
                pipe = pipe2;
            pipeline_graph.insert({pipe, {}});
        }

        for (auto &[key, _] : rel) {
            auto consumer = key.second;
            auto producer = key.first;

            if (consumer == pipe1) {
                consumer = pipe2;
            } else if (producer == pipe1) {
                producer = pipe2;
            }

            if (producer == consumer)
                continue;

            if (pipeline_graph.find(consumer) == pipeline_graph.end()) {
                pipeline_graph.insert({consumer, {producer}});
            } else {
                pipeline_graph.at(consumer).insert(producer);
            }
        }

        /*for (auto node : pipeline_graph) {
            llvm::outs() << "Key: " << node.first << ", Values: ";
            for (auto dependency : node.second) {
                llvm::outs() << dependency << " ";
            }
            llvm::outs() << "\n";
        }
        llvm::outs() << "\n";*/

        return tryTopologicalSort(pipeline_graph);
    }

    static std::map<PipelinePair, DisconnectReason>
    consolidateProducerConsumerRelationship(std::multimap<PipelinePair, DisconnectReason> mmPCR) {
        std::map<PipelinePair, DisconnectReason> pcr;
        for (const auto &[pipePair, disReason] : mmPCR) {
            if (pcr.find(pipePair) == pcr.end()) {
                pcr.insert({pipePair, disReason});
            } else {
                // Overwrite if INVALID as it domiantes MULTI_CONSUMER relationship
                if (disReason == DisconnectReason::INVALID) {
                    pcr.insert_or_assign(pipePair, disReason);
                }
            }
        }
        return pcr;
    }

    //------------------------------------------------------------------------------

  private:
    // kahn: https://dev.to/leopfeiffer/topological-sort-with-kahns-algorithm-3dl1
    // https://leetcode.com/problems/course-schedule/solutions/483330/c-kahns-algorithm-topological-sort-with-easy-detailed-explanation-16-ms-beats-98/
    static bool tryTopologicalSort(std::map<size_t, std::unordered_set<size_t>> pipeline_graph) {

        std::unordered_map<size_t, size_t> inDegrees;
        for (auto node : pipeline_graph) {
            for (auto dependency : node.second) {
                ++inDegrees[dependency];
            }
        }

        std::queue<size_t> queue;
        for (auto node : pipeline_graph) {
            if (inDegrees[node.first] == 0) {
                queue.push(node.first);
            }
        }

        std::vector<size_t> result;
        while (!queue.empty()) {
            size_t node = queue.front();
            queue.pop();
            result.push_back(node);
            for (auto dependency : pipeline_graph.at(node)) {
                if (--inDegrees[dependency] == 0) {
                    queue.push(dependency);
                }
            }
        }

        return result.size() == pipeline_graph.size();
    }

    static bool tryTopologicalSort(std::map<Pipeline *, std::unordered_set<Pipeline *>> pipeline_graph) {

        std::unordered_map<Pipeline *, size_t> inDegrees;
        for (auto node : pipeline_graph) {
            for (auto dependency : node.second) {
                ++inDegrees[dependency];
            }
        }

        std::queue<Pipeline *> queue;
        for (auto node : pipeline_graph) {
            if (inDegrees[node.first] == 0) {
                queue.push(node.first);
            }
        }

        std::vector<Pipeline *> result;
        while (!queue.empty()) {
            Pipeline *node = queue.front();
            queue.pop();
            result.push_back(node);
            for (auto dependency : pipeline_graph.at(node)) {
                if (--inDegrees[dependency] == 0) {
                    queue.push(dependency);
                }
            }
        }

        return result.size() == pipeline_graph.size();
    }

  public:
    /**
     * @brief Recursive function checking if the given value is transitively dependant on the operation `op`.
     * @param value The value to check
     * @param op The operation to check
     * @return true if there is a dependency, false otherwise
     */
    static bool valueDependsOnResultOf(mlir::Value value, mlir::Operation *op) {
        if (auto defOp = value.getDefiningOp()) {
            if (defOp == op)
                return true;
#if 1
            // TODO This crashes if defOp and op are not in the same block.
            // At the same time, it does not seem to be strictly required.
            //            if (defOp->isBeforeInBlock(op))
            // Nevertheless, this modified line seems to be a good soft-filter;
            // without that, the vectorization pass may take very long on
            // programs with 100s of operations.
            if (defOp->getBlock() == op->getBlock() && defOp->isBeforeInBlock(op))
                // can't have results of `op` as inputs, as it is defined before
                return false;
#endif
            for (auto operand : defOp->getOperands()) {
                if (valueDependsOnResultOf(operand, op))
                    return true;
            }
        }
        return false;
    }

    /**
     * @brief Moves operation which are between the operations, which should be fused into a single pipeline, before
     * or after the position where the pipeline will be placed.
     * @param pipelinePosition The position where the pipeline will be
     * @param pipeline The pipeline for which this function should be executed
     */
    static void movePipelineInterleavedOperations(mlir::Block::iterator pipelinePosition,
                                                  const std::vector<mlir::Operation *> pipeline) {
        // first operation in pipeline vector is last in IR, and the last is the first
        auto startPos = pipeline.back()->getIterator();
        auto endPos = pipeline.front()->getIterator();
        auto currSkip = pipeline.rbegin();

        std::vector<mlir::Operation *> moveBeforeOps;
        std::vector<mlir::Operation *> moveAfterOps;

        for (auto it = startPos; it != endPos; ++it) {
            if (it == (*currSkip)->getIterator()) {
                ++currSkip;
                continue;
            }

            bool dependsOnPipeline = false;
            auto pipelineOpsBeforeIt = currSkip;
            while (--pipelineOpsBeforeIt != pipeline.rbegin()) {
                for (auto operand : it->getOperands()) {
                    if (valueDependsOnResultOf(operand, *pipelineOpsBeforeIt)) {
                        dependsOnPipeline = true;
                        break;
                    }
                }
                if (dependsOnPipeline) {
                    break;
                }
            }

            for (auto operand : it->getOperands()) {
                if (valueDependsOnResultOf(operand, *pipelineOpsBeforeIt)) {
                    dependsOnPipeline = true;
                    break;
                }
            }
            if (dependsOnPipeline) {
                moveAfterOps.push_back(&(*it));
            } else {
                moveBeforeOps.push_back(&(*it));
            }
        }

        for (auto moveBeforeOp : moveBeforeOps) {
            moveBeforeOp->moveBefore(pipelinePosition->getBlock(), pipelinePosition);
        }
        for (auto moveAfterOp : moveAfterOps) {
            moveAfterOp->moveAfter(pipelinePosition->getBlock(), pipelinePosition);
            pipelinePosition = moveAfterOp->getIterator();
        }
    }

    static void createVectorizedPipelineOps(mlir::func::FuncOp func, std::vector<Pipeline> pipelines,
                                            std::map<mlir::Operation *, VectorIndex> decisionIxs) {
        mlir::OpBuilder builder(func);

        // Create the `VectorizedPipelineOp`s
        for (auto _pipeline : pipelines) {
            if (_pipeline.empty())
                continue;

            auto valueIsPartOfPipeline = [&](mlir::Value operand) {
                return llvm::any_of(_pipeline, [&](mlir::Operation *lv) { return lv == operand.getDefiningOp(); });
            };
            std::vector<mlir::Attribute> vSplitAttrs;
            std::vector<mlir::Attribute> vCombineAttrs;
            std::vector<mlir::Location> locations;
            std::vector<mlir::Value> results;
            std::vector<mlir::Value> operands;
            std::vector<mlir::Value> outRows;
            std::vector<mlir::Value> outCols;

            // first op in pipeline is last in IR
            builder.setInsertionPoint(_pipeline.front());
            // move all operations, between the operations that will be part of the pipeline, before or after the
            // completed pipeline
            VectorUtils::movePipelineInterleavedOperations(builder.getInsertionPoint(), _pipeline);

            // potential addition for
            std::vector<mlir::Operation *> pipeline;
            for (auto vIt = _pipeline.rbegin(); vIt != _pipeline.rend(); ++vIt) {
                auto v = *vIt;

                auto vSplits = std::vector<mlir::daphne::VectorSplit>();
                auto vCombines = std::vector<mlir::daphne::VectorCombine>();
                auto opsOutputSizes = std::vector<std::pair<mlir::Value, mlir::Value>>();
                if (auto vec = llvm::dyn_cast<mlir::daphne::Vectorizable>(v)) {
                    size_t d = decisionIxs[v];
                    vSplits = vec.getVectorSplits()[d];
                    vCombines = vec.getVectorCombines()[d];
                    opsOutputSizes = vec.createOpsOutputSizes(builder)[d];
                } else {
                    throw std::runtime_error("Vectorizable op not found");
                }

                pipeline.push_back(v);

                // TODO: although we do create enum attributes, it might make sense/make it easier to
                // just directly use an I64ArrayAttribute
                // Determination of operands of VectorizedPipelineOps!
                for (auto i = 0u; i < v->getNumOperands(); ++i) {
                    auto operand = v->getOperand(i);
                    if (!valueIsPartOfPipeline(operand)) {
                        vSplitAttrs.push_back(mlir::daphne::VectorSplitAttr::get(func.getContext(), vSplits[i]));
                        operands.push_back(operand);
                    }
                }

                // Determination of results of VectorizedPipelineOps!
                for (auto vCombine : vCombines) {
                    vCombineAttrs.push_back(mlir::daphne::VectorCombineAttr::get(func.getContext(), vCombine));
                }
                locations.push_back(v->getLoc());
                for (auto result : v->getResults()) {
                    results.push_back(result);
                }
                for (auto outSize : opsOutputSizes) {
                    outRows.push_back(outSize.first);
                    outCols.push_back(outSize.second);
                }

                // check if any of the outputs type of an operator is a scalar value
                // if yes, add additional castOps inside pipeline and outside pipeline
                for (size_t i = 0; i < v->getNumResults(); i++) {
                    auto r = v->getResult(0);
                    // TODO: check if it includes all types used in daphne
                    if (r.getType().isIntOrIndexOrFloat()) {
                        auto m1x1 = mlir::daphne::MatrixType::get(func.getContext(), r.getType(), 1, 1, 1,
                                                                  mlir::daphne::MatrixRepresentation::Dense);
                        auto loc = v->getLoc();

                        auto toCastOp = builder.create<mlir::daphne::CastOp>(loc, m1x1, r);
                        toCastOp->moveAfter(v);

                        // xxxxxx
                        pipeline.push_back(toCastOp);
                        vCombineAttrs.push_back(mlir::daphne::VectorCombineAttr::get(func.getContext(), vCombines[i]));
                        auto cst1 = builder.create<mlir::daphne::ConstantOp>(loc, builder.getIndexType(),
                                                                             builder.getIndexAttr(1l));
                        outRows.push_back(cst1);
                        outCols.push_back(cst1);
                        results.push_back(toCastOp);

                        auto fromCastOp = builder.create<mlir::daphne::CastOp>(loc, r.getType(), toCastOp);
                        r.replaceAllUsesExcept(fromCastOp, toCastOp);

                        mlir::Operation *firstUseOp = nullptr;
                        for (const auto &use : fromCastOp->getUses()) {
                            auto user = use.getOwner();

                            if (!firstUseOp || user->isBeforeInBlock(firstUseOp)) {
                                firstUseOp = user;
                            }
                        }

                        fromCastOp->moveBefore(firstUseOp);
                    }
                }
            }

            std::vector<mlir::Location> locs;
            locs.reserve(_pipeline.size());
            for (auto op : pipeline) {
                locs.push_back(op->getLoc());
            }

            auto loc = builder.getFusedLoc(locs);
            auto pipelineOp = builder.create<mlir::daphne::VectorizedPipelineOp>(
                loc, mlir::ValueRange(results).getTypes(), operands, outRows, outCols,
                builder.getArrayAttr(vSplitAttrs), builder.getArrayAttr(vCombineAttrs), nullptr);
            mlir::Block *bodyBlock = builder.createBlock(&pipelineOp.getBody());

            // remove information from input matrices of pipeline
            for (size_t i = 0u; i < operands.size(); ++i) {
                auto argTy = operands[i].getType();
                switch (vSplitAttrs[i].cast<mlir::daphne::VectorSplitAttr>().getValue()) {
                case mlir::daphne::VectorSplit::ROWS: {
                    auto matTy = argTy.cast<mlir::daphne::MatrixType>();
                    // only remove row information
                    argTy = matTy.withShape(-1, matTy.getNumCols());
                    break;
                }
                case mlir::daphne::VectorSplit::COLS: {
                    auto matTy = argTy.cast<mlir::daphne::MatrixType>();
                    // only remove col information
                    argTy = matTy.withShape(matTy.getNumRows(), -1);
                    break;
                }
                case mlir::daphne::VectorSplit::NONE:
                    // keep any size information
                    break;
                }
                bodyBlock->addArgument(argTy, builder.getUnknownLoc());
            }

            auto argsIx = 0u;
            auto resultsIx = 0u;
            // for every op in pipeline
            try {
            
            for (auto vIt = pipeline.begin(); vIt != pipeline.end(); ++vIt) {
                auto v = *vIt;
                auto numOperands = v->getNumOperands();
                auto numResults = v->getNumResults();

                // move v before end of block
                v->moveBefore(bodyBlock, bodyBlock->end());

                // set operands to arguments of body block, if defOp is not part of the pipeline
                for (auto i = 0u; i < numOperands; ++i) {
                    if (!valueIsPartOfPipeline(v->getOperand(i))) {
                        v->setOperand(i, bodyBlock->getArgument(argsIx++));
                    }
                }

                auto pipelineReplaceResults = pipelineOp->getResults().drop_front(resultsIx).take_front(numResults);
                resultsIx += numResults;
                for (auto z : llvm::zip(v->getResults(), pipelineReplaceResults)) {
                    auto old = std::get<0>(z);
                    auto replacement = std::get<1>(z);

                    // TODO: switch to type based size inference instead
                    // FIXME: if output is dynamic sized, we can't do this
                    // replace `NumRowOp` and `NumColOp`s for output size inference
                    for (auto &use : old.getUses()) {
                        
                        auto *op = use.getOwner();

                        if (auto nrowOp = llvm::dyn_cast<mlir::daphne::NumRowsOp>(op)) {
                            nrowOp.replaceAllUsesWith(pipelineOp.getOutRows()[replacement.getResultNumber()]);
                            nrowOp.erase();
                        }
                        if (auto ncolOp = llvm::dyn_cast<mlir::daphne::NumColsOp>(op)) {
                            ncolOp.replaceAllUsesWith(pipelineOp.getOutCols()[replacement.getResultNumber()]);
                            ncolOp.erase();
                        }
                    }
                    // Replace only if not used by pipeline op
                    old.replaceUsesWithIf(replacement, [&](mlir::OpOperand &opOperand) {
                        return llvm::count(pipeline, opOperand.getOwner()) == 0;
                    });
                }
            }
            } catch (...) {
                llvm::outs() << "TEST:" << "\n";
                func.print(llvm::outs());
                llvm::outs() << "\n";
            }
            bodyBlock->walk([](mlir::Operation *op) {
                for (auto resVal : op->getResults()) {
                    if (auto ty = resVal.getType().dyn_cast<mlir::daphne::MatrixType>()) {
                        resVal.setType(ty.withShape(-1, -1));
                    }
                }
            });
            builder.setInsertionPointToEnd(bodyBlock);
            builder.create<mlir::daphne::ReturnOp>(loc, results);
            if (!mlir::sortTopologically(bodyBlock)) {
                throw std::runtime_error("topoSort");
            }
        }
    }

    //-----------------------------------------------------------------
    //
    //-----------------------------------------------------------------

    struct DEBUG {

        static std::string getColor(size_t pipelineId) {
            std::vector<std::string> colors = {"tomato",   "lightgreen",  "lightblue",    "plum1",      "mistyrose2",
                                               "seashell", "hotpink",     "lemonchiffon", "firebrick1", "ivory2",
                                               "khaki1",   "lightcyan",   "olive",        "yellow",     "maroon",
                                               "violet",   "navajowhite1"};
            return colors[pipelineId % colors.size()];
        }

        static void drawPipelines(const std::vector<mlir::Operation *> &ops,
                                  const std::map<mlir::Operation *, size_t> &operationToPipelineIx,
                                  const std::map<mlir::Operation *, VectorIndex> &decisionIxs, std::string filename) {
            std::ofstream outfile(filename);

            outfile << "digraph G {" << std::endl;

            std::map<mlir::Operation *, std::string> opToNodeName;

            for (size_t i = 0; i < ops.size(); ++i) {
                std::string nodeName = "node" + std::to_string(i);
                opToNodeName[ops.at(i)] = nodeName;

                size_t pipelineId = operationToPipelineIx.at(ops[i]);
                VectorIndex vectIx = decisionIxs.at(ops.at(i));
                std::string color = VectorUtils::DEBUG::getColor(pipelineId);

                outfile << nodeName << " [label=\"" << ops.at(i)->getName().getStringRef().str()
                        << "\\npIx: " << pipelineId << ", vectIx: " << vectIx << "\", fillcolor=" << color
                        << ", style=filled];" << std::endl;
            }

            std::unordered_set<mlir::Operation *> outsideOps;

            for (size_t i = 0; i < ops.size(); ++i) {
                mlir::Operation *op = ops.at(i);
                auto consumerPipelineIx = operationToPipelineIx.at(op);

                for (const auto &operandValue : op->getOperands()) {
                    mlir::Operation *operandOp = operandValue.getDefiningOp();
                    auto it = operationToPipelineIx.find(operandOp);

                    if (it != operationToPipelineIx.end()) {
                        auto producerPipeplineIx = it->second;
                        outfile << opToNodeName.at(operandOp) << " -> " << opToNodeName.at(op);

                        if (producerPipeplineIx != consumerPipelineIx) {
                            outfile << " [style=dotted]";
                        }
                        outfile << ";" << std::endl;
                    } else {
                        // also show the surrounding ops, e.g. to make horizontal fusion visible
                    }
                }
            }
            outfile << "}" << std::endl;
        }

        static std::string printPtr(void *ptr) {

            std::ostringstream oss;
            oss << std::hex << reinterpret_cast<uintptr_t>(ptr);

            std::string str = oss.str();

            return str.substr(str.size() - 3);
        }

        static void drawPipelines(const std::vector<mlir::Operation *> &ops,
                                  const std::map<mlir::Operation *, Pipeline *> &operationToPipeline,
                                  const std::map<mlir::Operation *, VectorIndex> &decisionIxs, std::string filename) {
            std::ofstream outfile(filename);

            outfile << "digraph G {" << std::endl;

            std::map<mlir::Operation *, std::string> opToNodeName;
            std::map<Pipeline *, size_t> pipelineToIx;

            for (size_t i = 0; i < ops.size(); ++i) {
                std::string nodeName = "node" + std::to_string(i);
                opToNodeName[ops.at(i)] = nodeName;

                auto pipeline = operationToPipeline.at(ops.at(i));
                size_t pipelineIx;
                if (pipelineToIx.find(pipeline) == pipelineToIx.end()) {
                    pipelineIx = pipelineToIx.size();
                    pipelineToIx.insert({pipeline, pipelineIx});
                } else {
                    pipelineIx = pipelineToIx.at(pipeline);
                }
                std::string color = VectorUtils::DEBUG::getColor(pipelineIx);
                VectorIndex vectIx = decisionIxs.at(ops.at(i));

                std::string pipeName = printPtr(pipeline);

                outfile << nodeName << " [label=\"" << ops.at(i)->getName().getStringRef().str()
                        << "\\npIx: " << pipeName << ", vectIx: " << vectIx << "\", fillcolor=" << color
                        << ", style=filled];" << std::endl;
            }

            std::unordered_set<mlir::Operation *> outsideOps;

            for (size_t i = 0; i < ops.size(); ++i) {
                mlir::Operation *op = ops.at(i);
                auto consumerPipelineIx = operationToPipeline.at(op);

                for (const auto &operandValue : op->getOperands()) {
                    mlir::Operation *operandOp = operandValue.getDefiningOp();
                    auto it = operationToPipeline.find(operandOp);

                    if (it != operationToPipeline.end()) {
                        auto producerPipeplineIx = it->second;
                        outfile << opToNodeName.at(operandOp) << " -> " << opToNodeName.at(op);

                        if (producerPipeplineIx != consumerPipelineIx) {
                            outfile << " [style=dotted]";
                        }
                        outfile << ";" << std::endl;
                    } else {
                        // also show the surrounding ops, e.g. to make horizontal fusion visible
                    }
                }
            }
            outfile << "}" << std::endl;
        }

        static void drawPipelineOps(std::vector<mlir::daphne::VectorizedPipelineOp> &ops, std::string filename) {
            std::ofstream outfile(filename);

            outfile << "digraph GGroup {" << "\n";
            outfile << "compound=true;" << "\n";

            std::map<mlir::Operation *, std::string> opToNodeName;
            std::map<mlir::daphne::VectorizedPipelineOp, std::string> pipeOpToNodeName;
            std::map<mlir::Operation *, size_t> operationToPipeline;
            // std::map<mlir::Value, std::string> argToName;

            for (size_t i = 0; i < ops.size(); ++i) {
                std::string pipeName = "pipeOp" + std::to_string(i);
                pipeOpToNodeName.insert({ops.at(i), pipeName});

                std::string color = VectorUtils::DEBUG::getColor(i);

                outfile << "subgraph cluster_" << pipeName << " {\n";

                outfile << "label=\"S: [";
                for (const auto &x : ops.at(i).getSplits()) {
                    auto attr = static_cast<uint64_t>(llvm::dyn_cast<mlir::daphne::VectorSplitAttr>(x).getValue());
                    outfile << attr << ", ";
                }
                outfile << "]\\n";

                outfile << " C: [";
                for (const auto &x : ops.at(i).getCombines()) {
                    auto attr = static_cast<uint64_t>(llvm::dyn_cast<mlir::daphne::VectorCombineAttr>(x).getValue());
                    outfile << attr << ", ";
                }
                outfile << "]\";\n";

                outfile << "node [style=filled,color=" << color << "];\n";
                outfile << "color=" << "lightgrey" << ";\n";
                size_t j = 0;

                mlir::Block *b = &ops.at(i).getBody().getBlocks().front();

                for (const auto &arg : b->getArguments()) {
                    std::string argName = "arg" + std::to_string(arg.getArgNumber());
                    std::string qualArgName = pipeName + "_" + argName;
                    outfile << qualArgName << "[label=\"" << argName << "\"shape=diamond,color=grey];\n";
                    // argToName.insert({arg, qualArgName});
                }

                for (auto it = b->begin(); it != b->end(); ++it) {
                    mlir::Operation *op = &(*it);
                    std::string nodeName = pipeName + "_node" + std::to_string(j);
                    opToNodeName.insert({op, nodeName});
                    operationToPipeline.insert({op, i});
                    outfile << nodeName << " [label=\"" << op->getName().getStringRef().str() << "\"];\n";
                    j++;
                }
                outfile << pipeName << "_inv [style=invis,shape=point]" << ";\n";
                outfile << "}" << "\n";
            }

            for (size_t i = 0; i < ops.size(); ++i) {
                std::string pipeName = pipeOpToNodeName.at(ops.at(i));

                mlir::Block *b = &ops.at(i).getBody().getBlocks().front();
                for (auto it = b->begin(); it != b->end(); ++it) {
                    mlir::Operation *op = &(*it);

                    if (llvm::isa<mlir::daphne::ReturnOp>(op)) {
                        outfile << opToNodeName.at(op) << " -> " << pipeName << "_inv" << ";\n";
                    }

                    for (const auto &operandValue : op->getOperands()) {
                        auto operandOp = operandValue.getDefiningOp();
                        auto it = operationToPipeline.find(operandOp);

                        if (it != operationToPipeline.end()) {
                            outfile << opToNodeName.at(operandOp) << " -> " << opToNodeName.at(op);
                            outfile << ";" << std::endl;
                        } else {
                            if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(operandValue)) {
                                std::string argName = "arg" + std::to_string(arg.getArgNumber());
                                std::string qualArgName = pipeName + "_" + argName;
                                outfile << qualArgName << " -> " << opToNodeName.at(op) << ";\n";
                            }
                        }
                    }
                }
            }

            for (size_t i = 0; i < ops.size(); ++i) {
                std::string pipeName = pipeOpToNodeName.at(ops.at(i));
                auto op = ops.at(i);

                for (size_t j = 0; j < op.getSplits().size(); ++j) {
                    if (auto operandOp = op.getOperand(j).getDefiningOp()) {
                        if (auto defOp = llvm::dyn_cast<mlir::daphne::VectorizedPipelineOp>(operandOp)) {
                            std::string pipeName2 = pipeOpToNodeName.at(defOp);
                            std::string argName = pipeName + "_arg" + std::to_string(j);
                            outfile << pipeName2 << "_inv" << " -> " << argName << "[ltail=cluster_" << pipeName2
                                    << "];\n";
                        }
                    }
                }
            }
            outfile << "}" << "\n";
        }
    };
};