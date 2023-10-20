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

#ifdef USE_CUDA
#include "compiler/codegen/spoof-launcher/SpoofCUDAContext.h"
#include "compiler/utils/CompilerUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "runtime/local/context/CUDAContext.h"

#include <mlir/IR/IRMapping.h>

using namespace mlir;

static std::string cc_test_source = "TMP6\n"
                                    "\n"
                                    "21: // RowType: ROW_AGG\n"
                                    "22: // ConstDim2: -1\n"
                                    "23: // TB1: false\n"
                                    "24: // VectMem: 0\n"
                                    "25: \n"
                                    "26: #include \"agg_ops.cuh\"\n"
                                    "27: #include \"reduction.cuh\"\n"
                                    "28: #include \"spoof_utils.cuh\"\n"
                                    "29: #include \"utils.cuh\"\n"
                                    "30: #include \"Matrix.h\"\n"
                                    "31: #include \"TempStorage.cuh\"\n"
                                    "32: \n"
                                    "33: enum RowType {\n"
                                    "34:     NO_AGG_,       //no aggregation\n"
                                    "35:     NO_AGG_B1_,    //no aggregation w/ matrix mult B1\n"
                                    "36:     NO_AGG_CONST_, //no aggregation w/ expansion/contraction\n"
                                    "37:     FULL_AGG_,     //full row/col aggregation\n"
                                    "38:     ROW_AGG_,      //row aggregation (e.g., rowSums() or X %*% v)\n"
                                    "39:     COL_AGG_,      //col aggregation (e.g., colSums() or t(y) %*% X)\n"
                                    "40:     COL_AGG_T_,    //transposed col aggregation (e.g., t(X) %*% y)\n"
                                    "41:     COL_AGG_B1_,   //col aggregation w/ matrix mult B1\n"
                                    "42:     COL_AGG_B1_T_, //transposed col aggregation w/ matrix mult B1\n"
                                    "43:     COL_AGG_B1R_,  //col aggregation w/ matrix mult B1 to row vector\n"
                                    "44:     COL_AGG_CONST_ //col aggregation w/ expansion/contraction\n"
                                    "45: };\n"
                                    "46: \n"
                                    "47: \n"
                                    "48: template<typename T, int NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>\n"
                                    "49: struct SpoofRowwiseOp \n"
                                    "50: {\n"
                                    "51: \tMatrixAccessor<T> a;\n"
                                    "52: \tMatrixAccessor<T> b[NUM_B];\n"
                                    "53: \tMatrixAccessor<T> c;\n"
                                    "54: \tT* scalars;\n"
                                    "55: \tuint32_t grix;\n"
                                    "56: \tT* avals;\n"
                                    "57: \tuint32_t* aix;\n"
                                    "58: \tuint32_t alen;\n"
                                    "59: \n"
                                    "60: \tSpoofRowwiseOp(Matrix<T>* A, Matrix<T>* B, Matrix<T>* C, T* scalars, T* tmp_stor, uint32_t grix) :\n"
                                    "61: \t\t        scalars(scalars), grix(grix)  {\n"
                                    "62: \t\ta.init(A);\n"
                                    "63: \t\tc.init(C);\n"
                                    "64: \t\t\n"
                                    "65: \t\tif(B) {\n"
                                    "66: \t\t    for(auto i = 0; i < NUM_B; ++i)\n"
                                    "67: \t\t        b[i].init(&(B[i]));\n"
                                    "68: \t\t}\n"
                                    "69: \t}\n"
                                    "70: \n"
                                    "71: \t__device__  __forceinline__ void exec_dense(uint32_t ai, uint32_t ci, uint32_t rix) {\n"
                                    "72: \t\tT TMP3 = rowMaxsVectMult(a.vals(0), b[0].vals(0), ai, 0, a.cols());\n"
                                    "73: \t\tT TMP4 = getValue(b[1], rix);\n"
                                    "74: \t\tT TMP5 = max(TMP3, TMP4);\n"
                                    "75: \t\tif(tid == 0 || block_dim == 1){\n"
                                    "76: \t\t\t*(c.vals(rix)) = TMP5;\n"
                                    "77: \t\t}\n"
                                    "78: \n"
                                    "79: \t}\n"
                                    "80: \n"
                                    "81: \t__device__  __forceinline__ void exec_sparse(uint32_t ai, uint32_t ci, uint32_t rix, uint32_t tid, uint32_t block_dim) {\n"
                                    "82: \t\tT TMP3 = rowMaxsVectMult(avals, b[0].vals(0), aix, ai, 0, alen, tid, block_dim);\n"
                                    "83: \t\tT TMP4 = getValue(b[1], rix);\n"
                                    "84: \t\tT TMP5 = max(TMP3, TMP4);\n"
                                    "85: \t\tif(tid == 0 || block_dim == 1){\n"
                                    "86: \t\t\t*(c.vals(rix)) = TMP5;\n"
                                    "87: \t\t}\n"
                                    "88: \n"
                                    "89: \t}\n"
                                    "90: };\n"
                                    "91: \n"
                                    "92: \n"
                                    "93: template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>\n"
                                    "94: __global__ void TMP6_DENSE (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor,\n"
                                    "95:         uint32_t grix, uint32_t* bins, uint32_t bin_num, uint32_t bin_size)\n"
                                    "96: {\n"
                                    "97: \tconst uint& rix = blockIdx.x;\n"
                                    "98: \tSpoofRowwiseOp<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN> spoof_op(a, b, c, scalars, tmp_stor, grix + rix);\n"
                                    "99: \tspoof_op.exec_dense(rix * a->cols, rix * c->cols, rix);\n"
                                    "100: };\n"
                                    "101: \n"
                                    "102: \n"
                                    "103: template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>\n"
                                    "104: __device__ void exec_sparse(Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor, uint32_t grix, uint32_t rix,\n"
                                    "105:         uint32_t tid, uint32_t block_dim)\n"
                                    "106: {\n"
                                    "107:     SpoofRowwiseOp<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN> spoof_op(a, b, c, scalars, tmp_stor, grix + rix);\n"
                                    "108:     spoof_op.alen = spoof_op.a.row_len(rix);\n"
                                    "109:     spoof_op.aix = spoof_op.a.col_idxs(0);\n"
                                    "110:     spoof_op.avals = spoof_op.a.vals(0);\n"
                                    "111:     spoof_op.exec_sparse(a->row_ptr[rix], rix * c->cols, rix, tid, block_dim);\n"
                                    "112: }\n"
                                    "113: \n"
                                    "114: \n"
                                    "115: template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>\n"
                                    "116: __global__ void TMP6_SPARSE (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor,\n"
                                    "117:         uint32_t grix, uint32_t* bins, uint32_t bin_num, uint32_t bin_size)\n"
                                    "118: {\n"
                                    "119: \tconst uint32_t& rix = blockIdx.x;\n"
                                    "120:     exec_sparse<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN>(a, b, c, scalars, tmp_stor, grix, rix, threadIdx.x, blockDim.x);\n"
                                    "121: }\n"
                                    "122: \n"
                                    "123: \n"
                                    "124: template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>\n"
                                    "125: __global__ void TMP6_SPARSE_THREAD_BINS (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor,\n"
                                    "126:         uint32_t grix, uint32_t* bins, uint32_t bin_num, uint32_t bin_size) {\n"
                                    "127: \n"
                                    "128:     // global thread id\n"
                                    "129:     auto gtid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                                    "130: \n"
                                    "131:     if(gtid < bin_size) {\n"
                                    "132:         // bin index (either based on thread id for short rows (bin 0) or block id\n"
                                    "133:         const auto rix = bins[bin_num * a->rows + gtid];\n"
                                    "134: //        if(MatrixAccessor<T>(a).row_len(rix) > 0) {\n"
                                    "135: //        printf(\"gtid=%d < bin_size=%d; rix=%d\\n\", gtid, bin_size, rix);\n"
                                    "136:             exec_sparse<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN>(a, b, c, scalars, tmp_stor, grix, rix, gtid, 1);\n"
                                    "137: //        }\n"
                                    "138: //        else {\n"
                                    "139: //            RowType row_type = ROW_AGG_;\n"
                                    "140: //            auto ci  = rix * c->cols;\n"
                                    "141: //            if(row_type == NO_AGG_ || row_type == NO_AGG_CONST_) {\n"
                                    "142: //                auto i = 0;\n"
                                    "143: //                auto len = a->cols;\n"
                                    "144: //                while(i < len) {\n"
                                    "145: //                    c->data[ci+i++] = 0;\n"
                                    "146: //                }\n"
                                    "147: //            }\n"
                                    "148: //            else if(row_type == ROW_AGG_)\n"
                                    "149: //                c->data[rix] = 0;\n"
                                    "150: //            else if(row_type == FULL_AGG_)\n"
                                    "151: //                return;\n"
                                    "152: //            else\n"
                                    "153: //                printf(\"ERROR! row_type %d not handled in empty sparse row kernel\\n\", row_type);\n"
                                    "154: //        }\n"
                                    "155:     }\n"
                                    "156: }\n"
                                    "157: \n"
                                    "158: \n"
                                    "159: template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>\n"
                                    "160: __global__ void TMP6_SPARSE_BLOCK_BINS (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars,\n"
                                    "161:         T* tmp_stor, uint32_t grix, uint32_t* bins, uint32_t bin_num, uint32_t bin_size)\n"
                                    "162: {\n"
                                    "163:     // bin index (either based on thread id for short rows (bin 0) or block id\n"
                                    "164:     uint32_t bix = bin_num * a->rows + blockIdx.x;\n"
                                    "165:     const auto rix = bins[bix];\n"
                                    "166:     exec_sparse<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN>(a, b, c, scalars, tmp_stor, grix, rix, threadIdx.x, blockDim.x);\n"
                                    "167: }";

struct PatternBasedCodeGenPass : public PassWrapper<PatternBasedCodeGenPass, OperationPass<func::FuncOp>> {
    
    /**
     * @brief User configuration influencing the rewrite pass
     */
    const DaphneUserConfig& cfg;
    std::shared_ptr<spdlog::logger> logger;
    std::array<Operation*, 4> ccOpSequence;
    SpoofCUDAContext* codegenContext;

    explicit PatternBasedCodeGenPass(const DaphneUserConfig& cfg) : cfg(cfg) {
        logger = spdlog::get("compiler::cuda");
        ccOpSequence = std::array<Operation*, 4>({daphne::TransposeOp(), daphne::EwMulOp(), daphne::RowAggMaxOp(), daphne::EwMaxOp()});
        codegenContext = reinterpret_cast<SpoofCUDAContext*>(SpoofCUDAContext::initialize_cuda(0, "src/compiler/codegen"));
    }
    
    void runOnOperation() final;

};

void PatternBasedCodeGenPass::runOnOperation() {
    getOperation()->walk([&](Operation* op) {
        logger->debug("PatternBasedCodeGenPass: {} parent: {}", op->getName().getStringRef().str(),
                      op->getParentOp()->getName().getStringRef().str());
        OpBuilder builder(op);

//        auto cc_pattern = std::vector<Type>({daphne::TransposeOp(), daphne::EwMulOp(), daphne::RowAggMaxOp(), daphne::EwMaxOp()});

//        std::vector<Operation*> blab;
//        blab.push_back(Type<daphne::TransposeOp>);
//        blab.front()->dump();
//        if(auto blubb = llvm::dyn_cast<daphne::TransposeOp*>(blab.front()))
//            blubb->dump();

        if (auto constOp = llvm::dyn_cast<daphne::ConstantOp>(op))
        {
            logger->debug("constop");
            WalkResult::advance();
            return;
        }
//        else if (auto ccSeq = CompilerUtils::isOpSequence(op, ccOpSequence); ccSeq.size() == ccOpSequence.size()) {
//            logger->debug("found cc sequence");
//        }
        else if (auto ccSeq = CompilerUtils::isCCseq(op, ccOpSequence); ccSeq.size() == ccOpSequence.size()) {
            logger->debug("is cc sequence");

            std::vector<Location> locations;
            std::vector<Value> results;
            std::vector<Value> operands;

            for(auto seqOp: ccSeq) {
//                for(auto i = 0u; i < seqOp->getNumOperands(); ++i) {
//                    auto operand = seqOp->getOperand(i);
//                    operands.push_back(operand);
//                }
            locations.push_back(seqOp->getLoc());
//            for(auto result: pipelineOp->getResults()) {
//                results.push_back(result);
//            }
            }
            auto loc = builder.getFusedLoc(locations);
            auto result = ccSeq.back()->getResult(0);
            results.push_back(result);
//            for(auto j = 0u; j < 2; j++) {
//                ccSeq[j]->dump();
//                for (auto i = 0u; i < ccSeq[j]->getNumOperands(); ++i) {
//                    auto operand = op->getOperand(i);
//                    operands.push_back(operand);
//                }
//            }
            operands.push_back(ccSeq[1]->getOperand(0));
            operands.push_back(ccSeq[0]->getOperand(0));
            // ToDo: implement this functionality sketched here in pseudo code
            // std::string ops_source;
            // for(auto o : ccSeq) {
            //   ops_source += o->generateSource();
            // }
            // auto template_source = generateTemplateSource(ops_source);

            std::unique_ptr<SpoofRowwiseOp> cc_op = std::make_unique<SpoofRowwiseOp>(SpoofOperator::RowType(), false, 0, 0);
            auto opID = codegenContext->compile(std::move(cc_op), cc_test_source);

            operands.push_back(builder.create<daphne::ConstantOp>(loc, opID));
            auto generatedOp = builder.create<daphne::CodeGenOpRowwise>(loc, ValueRange(results).getTypes(), operands, nullptr);

            ccSeq.back()->getNextNode()->getOperand(0).replaceAllUsesWith(generatedOp.getResult(0));
//            for (auto it = ccSeq.rbegin(); it != ccSeq.rend(); ++it) {
//                (*it)->erase();
//            }
        }
        else {

            if(auto transOp = llvm::dyn_cast<daphne::TransposeOp>(op))
//                    llvm::dyn_cast<daphne::RowAggMaxOp>(op->use_begin()->get().use_begin()->get().getType()))
            {
//                if(llvm::dyn_cast<decltype(blabb)>(op))
//                daphne::TransposeOp top;
//                if(llvm::dyn_cast<decltype(top)>(op))
//                if(typeid(blabb[0]) == typeid(op))
//                    logger->debug("typeid: {}", typeid(blabb[0]).name());

                auto nextOp1 =  op->use_begin()->get().getDefiningOp();

                auto isEwMulOp = llvm::dyn_cast<daphne::EwMulOp>(nextOp1);
                auto nextOp2 = nextOp1->use_begin()->get().getDefiningOp();
                auto isRowAggMaxOp =  llvm::dyn_cast<daphne::RowAggMaxOp>(nextOp2);
                nextOp1 = nextOp2->use_begin()->get().getDefiningOp();
                auto isMaxOp =  llvm::dyn_cast<daphne::RowAggMaxOp>(nextOp1);

                if(isEwMulOp && isRowAggMaxOp && isMaxOp) {
                    logger->debug("Found sequence T->EwMul->RowAggMax->Max");
                }

//                auto defOp1 = transOp->getOperand(0).getDefiningOp();
//                defOp1->dump();
                logger->debug("{} num operands: {}", transOp->getName().getStringRef().str(), transOp->getNumOperands());
                auto ow1 = op->getUses().begin()->getOwner();
                ow1->dump();
                auto o1 = op->getOpOperands().front().get();
                o1.dump();
                auto o2 = op->getOperand(0);
                o2.dump();
                auto odefo = o2.getDefiningOp();
                auto opb = o2.getParentBlock();
                if(opb) {
                    opb->dump();
                    if(auto bla = opb->getParentOp())
                        bla->dump();
                    if(auto blar = o2.getParentRegion())
                        blar->getParentOp()->dump();
                }
                if(odefo)
                    odefo->dump();

                if (ow1) {
                    auto tmp1 = llvm::dyn_cast<daphne::EwMulOp>(ow1);
                    if (tmp1) {
                        logger->debug("ewmul suxessor: {}", tmp1->getUses().begin()->getOwner()->getName().getStringRef().str());
                        auto bla = tmp1->getUsers().begin();
                        auto rowaggop = llvm::dyn_cast<daphne::RowAggMaxOp>(bla->use_begin()->get().getDefiningOp());
                        rowaggop->getName().dump();
                        auto bla1 = tmp1->getNextNode();
                        bla1->dump();
                    }
                }
                auto sc = 0;
                for (auto s: op->getSuccessors()) {
                    logger->debug("Successor {}:", sc++);
                    s->dump();
                }

                logger->debug("operands:");
                for (const auto &o: op->getOpOperands()) {
                    o.get().dump();
                }

                logger->debug("definers:");
                for (const auto &o: op->getOpOperands()) {
                    if (o.get().getDefiningOp())
                        o.get().getDefiningOp()->dump();
                    else
                        logger->debug("no defining op in {}", op->getName().getStringRef().str());
                }

                logger->debug("users:");
                for (auto u: op->getUsers()) {
                    if (u)
                        u->dump();
                    else
                        logger->debug("no users of {}", op->getName().getStringRef().str());
                }
            }
        }
        WalkResult::advance();
    });
}

std::unique_ptr<Pass> daphne::createPatternBasedCodeGenPass(const DaphneUserConfig& cfg) {
    return std::make_unique<PatternBasedCodeGenPass>(cfg);
}

#endif // USE_CUDA