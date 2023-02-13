#include "ir/daphneir/Daphne.h"
#include "api/cli/DaphneUserConfig.h"
#include "ir/daphneir/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "runtime/local/context/DaphneContext.h"

#include "ir/daphneir/Daphne.h"

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input hello file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int dumpLLVMIR(mlir::ModuleOp module) {
    mlir::registerLLVMDialectTranslation(*module->getContext());
    // Convert the module to LLVM IR in a new LLVM IR context.

    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    // Optionally run an optimization pipeline over the llvm module.
    auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
        llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }
    llvm::outs() << *llvmModule << "\n";
    return 0;
}

int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }
    return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module) {
    if (int error = loadMLIR(context, module)) {
        return error;
    }

    // Register passes to be applied in this compile process
    mlir::PassManager passManager(&context);
    mlir::applyPassManagerCLOptions(passManager);

    DaphneUserConfig cfg{};
    // TODO
    // add daphne context creation pass

    passManager.addPass(mlir::daphne::createLowerDenseMatrixPass());
    module->dump();
    // passManager.addPass(mlir::createLowerAffinePass());
    // passManager.addNestedPass<mlir::FuncOp>(mlir::daphne::createInsertDaphneContextPass(cfg));
    // passManager.addPass(mlir::daphne::createLowerDenseMatrixPass());
    // passManager.addNestedPass<mlir::FuncOp>(mlir::daphne::createRewriteToCallKernelOpPass());

    // passManager.addPass(mlir::createCanonicalizerPass());
    // passManager.addPass(mlir::createCSEPass());
    // passManager.addNestedPass<mlir::FuncOp>(mlir::daphne::createRewriteToCallKernelOpPass());
    // passManager.addPass(mlir::createLowerToCFGPass());
    // passManager.addPass(mlir::daphne::createLowerToLLVMPass(cfg));

    if (mlir::failed(passManager.run(*module))) {
        return 4;
    }

    return 0;
}

int runJit(mlir::ModuleOp module) {
    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Register the translation from MLIR to LLVM IR, which must happen before
    // we can JIT-compile.
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // An optimization pipeline to use within the execution engine.
    unsigned make_fast = 0;
    auto optPipeline = mlir::makeOptimizingTransformer(make_fast, 0, nullptr);

    mlir::ExecutionEngineOptions options;
    options.llvmModuleBuilder = nullptr;
    options.transformer = optPipeline;
    options.jitCodeGenOptLevel = llvm::CodeGenOpt::Level::Default;
    options.enableObjectDump = true;
    options.enableGDBNotificationListener = true;
    options.enablePerfNotificationListener = true;

    // Create an MLIR execution engine. The execution engine eagerly
    // JIT-compiles the module.
    auto maybeEngine = mlir::ExecutionEngine::create(module, options);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    // Invoke the JIT-compiled function.
    auto invocationResult = engine->invokePacked("main");
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }

    return 0;
}

int main(int argc, char **argv) {
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "Hello compiler\n");
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::daphne::DaphneDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::AffineDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (int error = loadAndProcessMLIR(context, module)) {
        return error;
    }

    dumpLLVMIR(*module);
    //  runJit(*module);

    return 0;
}
