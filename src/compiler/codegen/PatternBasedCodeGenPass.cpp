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

static std::string cc_row_test_source = "TMP6\n"
                                    "\n"
                                    "// RowType: ROW_AGG\n"
                                    "// ConstDim2: -1\n"
                                    "// TB1: false\n"
                                    "// VectMem: 0\n"
                                    "\n"
                                    "#include \"agg_ops.cuh\"\n"
                                    "#include \"reduction.cuh\"\n"
                                    "#include \"spoof_utils.cuh\"\n"
                                    "#include \"utils.cuh\"\n"
                                    "#include \"Matrix.h\"\n"
                                    "#include \"TempStorage.cuh\"\n"
                                    "\n"
                                    "enum RowType {\n"
                                    "    NO_AGG_,       //no aggregation\n"
                                    "    NO_AGG_B1_,    //no aggregation w/ matrix mult B1\n"
                                    "    NO_AGG_CONST_, //no aggregation w/ expansion/contraction\n"
                                    "    FULL_AGG_,     //full row/col aggregation\n"
                                    "    ROW_AGG_,      //row aggregation (e.g., rowSums() or X %*% v)\n"
                                    "    COL_AGG_,      //col aggregation (e.g., colSums() or t(y) %*% X)\n"
                                    "    COL_AGG_T_,    //transposed col aggregation (e.g., t(X) %*% y)\n"
                                    "    COL_AGG_B1_,   //col aggregation w/ matrix mult B1\n"
                                    "    COL_AGG_B1_T_, //transposed col aggregation w/ matrix mult B1\n"
                                    "    COL_AGG_B1R_,  //col aggregation w/ matrix mult B1 to row vector\n"
                                    "    COL_AGG_CONST_ //col aggregation w/ expansion/contraction\n"
                                    "};\n"
                                    "\n"
                                    "\n"
                                    "template<typename T, int NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>\n"
                                    "struct SpoofRowwiseOp \n"
                                    "{\n"
                                    "\tMatrixAccessor<T> a;\n"
                                    "\tMatrixAccessor<T> b[NUM_B];\n"
                                    "\tMatrixAccessor<T> c;\n"
                                    "\tT* scalars;\n"
                                    "\tuint32_t grix;\n"
                                    "\tT* avals;\n"
                                    "\tsize_t* aix;\n"
                                    "\tuint32_t alen;\n"
                                    "\n"
                                    "\tSpoofRowwiseOp(CCMatrix<T>* A, CCMatrix<T>* B, CCMatrix<T>* C, T* scalars, T* tmp_stor, uint32_t grix) :\n"
                                    "\t\t        scalars(scalars), grix(grix)  {\n"
                                    "\t\ta.init(A);\n"
                                    "\t\tc.init(C);\n"
                                    "\t\t\n"
                                    "\t\tif(B) {\n"
                                    "\t\t    for(auto i = 0; i < NUM_B; ++i)\n"
                                    "\t\t        b[i].init(&(B[i]));\n"
                                    "\t\t}\n"
                                    "\t}\n"
                                    "\n"
                                    "\t__device__  __forceinline__ void exec_dense(uint32_t ai, uint32_t ci, uint32_t rix) {\n"
                                    "\tif(threadIdx.x == 0 && rix == 0) {\n"
                                    "\t\t//printf(\"rix=%d a->rows=%d b->cols=%d c->rows=%d\\n\", rix, a.rows(), b[0].cols(), c.rows());\n"
                                    "\t\t//printf(\"cuda sizeof(uint32_t*)=%l\\n\", sizeof(uint32_t*));\n"
                                    "\t}\n"
                                    "\t//return;\n"
                                    "\t\tT TMP3 = rowMaxsVectMult(a.vals(0), b[0].vals(0), ai, 0, a.cols());\n"
                                    "\t\tT TMP4 = getValue(b[0], rix);\n"
                                    "\t\tT TMP5 = max(TMP3, TMP4);\n"
                                    "\t\tauto tid = threadIdx.x;\n"
                                    "\t\tif(tid == 0) {\n"
                                    "\t\t\t//printf(\"tid=%d; bid=%d TMP3=%f; TMP4=%f; TMP5=%f\\n\", tid, blockIdx.x, TMP3, TMP4, TMP5);\n"
                                    "\t\t\t*(c.vals(rix)) = TMP5;\n"
                                    "\t\t}\n"
                                    "\n"
                                    "\t}\n"
                                    "\n"
                                    "\t__device__  __forceinline__ void exec_sparse(uint32_t ai, uint32_t ci, uint32_t rix, uint32_t tid, uint32_t block_dim) {\n"
                                        "\t//if(threadIdx.x == 0){// && rix == 4) {\n"
                                        "\t\t//printf(\"rix=%d a->rows=%d b->cols=%d c->rows=%d c->cols=%d row len/nnz=%d\\n\", rix, a.rows(), b[0].cols(), c.rows(), c.cols(), alen);\n"
                                        "//printf(\"rix=%d row_ptr=%d\\n\", rix, this->a.pos(rix))\n;"
                                        "\t//}\n"
                                        "//if(rix != 4 || threadIdx.x > 0)\n"
                                        "//\treturn;\n"
                                    "\t\tT TMP3 = rowMaxsVectMult(avals, b[0].vals(0), aix, ai, 0, alen, tid, block_dim);\n"
                                    "\t\tT TMP4 = getValue(b[0], rix);\n"
                                    "\t\tT TMP5 = max(TMP3, TMP4);\n"
                                    "\t\tif(tid == 0){ // || block_dim == 1){\n"
                                    "\t\t\t//printf(\"tid=%d; bid=%d TMP3=%f; TMP4=%f; TMP5=%f\\n\", tid, blockIdx.x, TMP3, TMP4, TMP5);\n"
                                    "\t\t\t*(c.vals(rix)) = TMP5;\n"
                                    "\t\t}\n"
                                    "\n"
                                    "\t}\n"
                                    "};\n"
                                    "\n"
                                    "\n"
                                    "template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>\n"
                                    "__global__ void TMP6_DENSE (CCMatrix<T>* a, CCMatrix<T>* b, CCMatrix<T>* c, T* scalars, T* tmp_stor,\n"
                                    "        uint32_t grix, uint32_t* bins, uint32_t bin_num, uint32_t bin_size)\n"
                                    "{\n"
                                    "\tconst uint& rix = blockIdx.x;\n"
                                    "\tSpoofRowwiseOp<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN> spoof_op(a, b, c, scalars, tmp_stor, grix + rix);\n"
                                    "\tspoof_op.exec_dense(rix * a->cols, rix * c->cols, rix);\n"
                                    "};\n"
                                    "\n"
                                    "\n"
                                    "template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>\n"
                                    "__device__ void exec_sparse(CCMatrix<T>* a, CCMatrix<T>* b, CCMatrix<T>* c, T* scalars, T* tmp_stor, uint32_t grix, uint32_t rix,\n"
                                    "        uint32_t tid, uint32_t block_dim)\n"
                                    "{\n"
                                    "     SpoofRowwiseOp<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN> spoof_op(a, b, c, scalars, tmp_stor, grix + rix);\n"
                                    "     spoof_op.alen = spoof_op.a.row_len(rix);\n"
                                    "     spoof_op.aix = spoof_op.a.col_idxs(0);\n"
                                    "     spoof_op.avals = spoof_op.a.vals(0);\n"
                                    "     spoof_op.exec_sparse(a->row_ptr[rix], rix * c->cols, rix, tid, block_dim);\n"
                                    " }\n"
                                    " \n"
                                    " \n"
                                    " template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>\n"
                                    " __global__ void TMP6_SPARSE (CCMatrix<T>* a, CCMatrix<T>* b, CCMatrix<T>* c, T* scalars, T* tmp_stor,\n"
                                    "         uint32_t grix, uint32_t* bins, uint32_t bin_num, uint32_t bin_size)\n"
                                    " {\n"
                                    " \tconst uint32_t& rix = blockIdx.x;\n"
                                    "     exec_sparse<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN>(a, b, c, scalars, tmp_stor, grix, rix, threadIdx.x, blockDim.x);\n"
                                    " }\n"
                                    " \n"
                                    " \n"
                                    " template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>\n"
                                    " __global__ void TMP6_SPARSE_THREAD_BINS (CCMatrix<T>* a, CCMatrix<T>* b, CCMatrix<T>* c, T* scalars, T* tmp_stor,\n"
                                    "         uint32_t grix, uint32_t* bins, uint32_t bin_num, uint32_t bin_size) {\n"
                                    " \n"
                                    "     // global thread id\n"
                                    "     auto gtid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                                    " \n"
                                    "    if(gtid < bin_size) {\n"
                                    "         // bin index (either based on thread id for short rows (bin 0) or block id\n"
                                    "         const auto rix = bins[bin_num * a->rows + gtid];\n"
                                    " //        if(MatrixAccessor<T>(a).row_len(rix) > 0) {\n"
                                    " //        printf(\"gtid=%d < bin_size=%d; rix=%d\\n\", gtid, bin_size, rix);\n"
                                    "             exec_sparse<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN>(a, b, c, scalars, tmp_stor, grix, rix, gtid, 1);\n"
                                    " //        }\n"
                                    " //        else {\n"
                                    " //            RowType row_type = ROW_AGG_;\n"
                                    " //            auto ci  = rix * c->cols;\n"
                                    " //            if(row_type == NO_AGG_ || row_type == NO_AGG_CONST_) {\n"
                                    " //                auto i = 0;\n"
                                    " //                auto len = a->cols;\n"
                                    " //                while(i < len) {\n"
                                    " //                    c->data[ci+i++] = 0;\n"
                                    " //                }\n"
                                    " //            }\n"
                                    " //            else if(row_type == ROW_AGG_)\n"
                                    " //                c->data[rix] = 0;\n"
                                    " //            else if(row_type == FULL_AGG_)\n"
                                    " //                return;\n"
                                    " //            else\n"
                                    " //                printf(\"ERROR! row_type %d not handled in empty sparse row kernel\\n\", row_type);\n"
                                    " //        }\n"
                                    "     }\n"
                                    " }\n"
                                    " \n"
                                    " \n"
                                    " template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>\n"
                                    " __global__ void TMP6_SPARSE_BLOCK_BINS (CCMatrix<T>* a, CCMatrix<T>* b, CCMatrix<T>* c, T* scalars,\n"
                                    "         T* tmp_stor, uint32_t grix, uint32_t* bins, uint32_t bin_num, uint32_t bin_size)\n"
                                    " {\n"
                                    "     // bin index (either based on thread id for short rows (bin 0) or block id\n"
                                    "     uint32_t bix = bin_num * a->rows + blockIdx.x;\n"
                                    "     const auto rix = bins[bix];\n"
                                    "     exec_sparse<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN>(a, b, c, scalars, tmp_stor, grix, rix, threadIdx.x, blockDim.x);\n"
                                    " }";

static std::string cc_cell_test_source = "TMP2\n"
                                     "// CellType: FULL_AGG\n"
                                     "// AggOp: SUM\n"
                                     "// SparseSafe: false\n"
                                     "// SEQ: false\n"
                                     "\n"
                                     "#include \"agg_ops.cuh\"\n"
                                     "#include \"reduction.cuh\"\n"
                                     "#include \"spoof_utils.cuh\"\n"
                                     "#include \"utils.cuh\"\n"
                                     "#include \"Matrix.h\"\n"
                                     "\n"
                                     "template<typename T, int NUM_B>\n"
                                     "struct SpoofCellwiseOp {\n"
                                     "\t\tMatrixAccessor<T> A;\n"
                                     "\t\tMatrixAccessor<T> b[NUM_B];\n"
                                     "\t\tMatrixAccessor<T> c;\n"
                                     "\t\tT* scalars;\n"
                                     "\t\tT* avals;\n"
                                     "\t\tsize_t* aix;\n"
                                     "\t\tuint32_t alen;\n"
                                     "\t\tuint32_t& n;\n"
                                     "\t\tuint32_t _grix;\n"
                                     "\n"
                                     "\tSpoofCellwiseOp(CCMatrix<T>* _A, CCMatrix<T>* _B, CCMatrix<T>* _C, T* scalars, uint32_t grix) :\n"
                                     "\t\tn(_A->cols), scalars(scalars), _grix(grix)\n"
                                     "\t{\n"
                                     "\t\tA.init(_A);\n"
                                     "\t\tc.init(_C);\n"
                                     "\t\talen = A.row_len(grix);\n"
                                     "\n"
                                     "\t\tif(_B)\n"
                                     "\t\t\tfor(auto i = 0; i < NUM_B; ++i)\n"
                                     "\t\t\t\tb[i].init(&(_B[i]));\n"
                                     "\t}\n"
                                     "\n"
                                     "\t__device__  __forceinline__ T operator()(T a, uint32_t idx, uint32_t rix, uint32_t cix) {\n"
                                     "\t\tT TMP0 = getValue(b[0], rix);\n"
                                     "\t\tT TMP1 = (a != TMP0) ? 1.0 : 0.0;\n"
                                     "\n"
                                     "\t\treturn TMP1;\n"
                                     "\t}\n"
                                     "};\n"
                                     "\n"
                                     "template<typename T, int NUM_B>\n"
                                     "__global__ void TMP2_DENSE (CCMatrix<T>* a, CCMatrix<T>* b, CCMatrix<T>* c, T* scalars, uint32_t n, uint32_t grix) {\n"
                                     "\t//IdentityOp<T> agg_op;\n"
                                     "\tSumOp<T> agg_op;"
                                     "\tSpoofCellwiseOp<T, NUM_B> spoof_op(a, b, c, scalars, grix);\n"
                                     "\t//NO_AGG<T, IdentityOp<T>, SpoofCellwiseOp<T, NUM_B>>(&(spoof_op.A), &(spoof_op.c), n, (T)1.0, agg_op, spoof_op);\n"
                                     "\tFULL_AGG<T, SumOp<T>, SpoofCellwiseOp<T, NUM_B>>(&(spoof_op.A), &(spoof_op.c), n, (T)0.0, agg_op, spoof_op);\n"
                                     "};\n"
                                     "\n"
                                     "template<typename T, int NUM_B>\n"
                                     "__global__ void TMP2_SPARSE (CCMatrix<T>* a, CCMatrix<T>* b, CCMatrix<T>* c, T* scalars, uint32_t n, uint32_t grix) {\n"
                                     "\tIdentityOp<T> agg_op;\n"
                                     "\tSpoofCellwiseOp<T, NUM_B> spoof_op(a, b, c, scalars, grix);\n"
                                     "\tNO_AGG_SPARSE<T, IdentityOp<T>, SpoofCellwiseOp<T, NUM_B>>(&(spoof_op.A), &(spoof_op.c), n, (T)1.0, agg_op, spoof_op);\n"
                                     "};";

struct PatternBasedCodeGenPass : public PassWrapper<PatternBasedCodeGenPass, OperationPass<func::FuncOp>> {
    
    /**
     * @brief User configuration influencing the rewrite pass
     */
    DaphneUserConfig& cfg;
    std::shared_ptr<spdlog::logger> logger;
    std::array<Operation*, 4> ccOpSequenceRW{};
    std::array<Operation*, 2> ccOpSequenceCW{};
    SpoofCUDAContext* codegenContext;

    explicit PatternBasedCodeGenPass(DaphneUserConfig& cfg) : cfg(cfg) {
        logger = spdlog::get("compiler::cuda");
        ccOpSequenceRW = std::array<Operation*, 4>({daphne::TransposeOp(), daphne::EwMulOp(), daphne::RowAggMaxOp(), daphne::EwMaxOp()});
        ccOpSequenceCW = std::array<Operation*, 2>({daphne::EwNegOp(), daphne::AllAggSumOp()});
        std::string ccg_resource_dir;
        auto daphne_root = std::getenv("DAPHNE_ROOT");
        if(daphne_root)
            ccg_resource_dir = std::string(daphne_root) + std::string("/src/compiler/codegen");
        else
            ccg_resource_dir = std::string("src/compiler/codegen");

        logger->debug("ccg resoruce dir: {}", ccg_resource_dir);
        codegenContext = reinterpret_cast<SpoofCUDAContext*>(SpoofCUDAContext::initialize_cuda(0, ccg_resource_dir.c_str()));
        cfg.codegen_ctx_ptr = reinterpret_cast<uint64_t>(codegenContext);
    }
    
    void runOnOperation() final;

};

void PatternBasedCodeGenPass::runOnOperation() {
    std::vector<Operation*> ops_to_remove;
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
        else if (auto ccSeq = CompilerUtils::isCCseqRW(op, ccOpSequenceRW); ccSeq.size() == ccOpSequenceRW.size()) {
            logger->debug("is cc rowwise sequence");

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
            cc_op->name = "TMP6";
            auto opID = codegenContext->compile(std::move(cc_op), cc_row_test_source);

            operands.push_back(builder.create<daphne::ConstantOp>(loc, opID));
            auto generatedOp = builder.create<daphne::CodeGenOpRowwise>(loc, ValueRange(results).getTypes(), operands, nullptr);

            ccSeq.back()->getNextNode()->getOperand(0).replaceAllUsesWith(generatedOp.getResult(0));
//            for (auto it = ccSeq.rbegin(); it != ccSeq.rend(); ++it) {
            for (auto it = ccSeq.begin(); it != ccSeq.end(); ++it) {

//                (*it)->erase();
                std::string s;
                llvm::raw_string_ostream stream(s);
                (*it)->print(stream);
                logger->debug("Queuing for deletion: {}", stream.str());
                ops_to_remove.push_back((*it));
            }
        }
        else if (auto ccSeqCW = CompilerUtils::isCCseqCW(op, ccOpSequenceCW); ccSeqCW.size() == ccOpSequenceCW.size()) {
//            WalkResult::advance();
//            return;
            logger->debug("is cc cellwise sequence");

            std::vector<Location> locations;
            std::vector<Value> results;
            std::vector<Value> operands;

            for(auto seqOp: ccSeqCW) {
                locations.push_back(seqOp->getLoc());
            }
            auto loc = builder.getFusedLoc(locations);
            auto result = ccSeqCW.back()->getResult(0);
            results.push_back(result);
//            operands.push_back(ccSeqCW[1]->getOperand(0));
            operands.push_back(ccSeqCW[0]->getOperand(0));
            operands.push_back(ccSeqCW[0]->getOperand(1));

            std::unique_ptr<SpoofCellwiseOp> cc_op = std::make_unique<SpoofCellwiseOp>(SpoofOperator::AggType::FULL_AGG, SpoofOperator::AggOp::SUM, false);
            cc_op->name = "TMP2";
            auto opID = codegenContext->compile(std::move(cc_op), cc_cell_test_source);

            operands.push_back(builder.create<daphne::ConstantOp>(loc, opID));
            auto generatedOp = builder.create<daphne::CodeGenOpAllAggCellwise>(loc, ValueRange(results).getTypes(), operands, nullptr);

//            logger->debug("replacing");
//            ccSeqCW.back()->dump();
//            logger->debug("with: ");
            ccSeqCW.back()->replaceAllUsesWith(generatedOp);

//            ccSeqCW.back()->getNextNode()->getOperand(0).replaceAllUsesWith(generatedOp.getResult(0));
//            for(auto arg : ccSeqCW.back()->getParentRegion()->getArguments()) {
//                arg.dump();
//                arg.replaceAllUsesWith(generatedOp.getResult(0));
//            }
//            ccSeqCW.back()->getNextNode()->getNextNode()->getOperand(0).replaceAllUsesWith(generatedOp.getResult(0));
//            for(auto o = 0; o < ccSeqCW.back()->getNumOperands(); o++) {
//                ccSeqCW.back()->getOperand(o).dump();
//            }

//            for (auto it = ccSeqCW.rbegin(); it != ccSeqCW.rend(); ++it) {
            for (auto it = ccSeqCW.begin(); it != ccSeqCW.end(); ++it) {
                std::string s;
                llvm::raw_string_ostream stream(s);
                (*it)->print(stream);
                logger->debug("Queuing for deletion: {}", stream.str());
                ops_to_remove.push_back((*it));
            }
        }
        else {
            WalkResult::advance();
            return;
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

    for (auto it = ops_to_remove.rbegin(); it != ops_to_remove.rend(); ++it) {
//        (*it)->erase();
//    }
//    for (auto it = ops_to_remove.begin(); it != ops_to_remove.end(); ++it) {
    //        std::string s;
    //        llvm::raw_string_ostream stream(s);
    //        (*it)->print(stream);
    //        logger->debug("Deleting {}", stream.str());
    //        logger->debug("users of {}:", stream.str());
    //        for (auto u: (*it)->getUsers()) {
    //            if (u)
    //                u->dump();
    //            else
    //                logger->debug("no users of {}", (*it)->getName().getStringRef().str());
    //        }
        (*it)->erase();
    }
}

std::unique_ptr<Pass> daphne::createPatternBasedCodeGenPass(DaphneUserConfig& cfg) {
    return std::make_unique<PatternBasedCodeGenPass>(cfg);
}

#endif // USE_CUDA