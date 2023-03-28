/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/


#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"


using namespace mlir;


struct KernelSelectionPass : public PassWrapper<KernelSelectionPass, OperationPass<func::FuncOp>> {
    
    const DaphneUserConfig& cfg;
    
    template<class OP>
    void setAPIifType(const std::string& API, OpBuilder builder, Operation * op){
        if(auto flagged_op = llvm::dyn_cast<OP>(op)){
                op->setAttr("API", builder.getStringAttr(API));
        }
    }
    
    KernelSelectionPass(const DaphneUserConfig& cfg) : cfg(cfg) {
        //
    }
    
    
    void runOnOperation() final;
    
};


void KernelSelectionPass::runOnOperation() {
    using namespace mlir::daphne;
    if(cfg.api == "MorphStore") {
        getOperation()->walk(
          [&](Operation * op) {
              OpBuilder builder(op);
          
              setAPIifType<EwEqOp>(cfg.api, builder, op);
              setAPIifType<EwNeqOp>(cfg.api, builder, op);
              setAPIifType<EwLtOp>(cfg.api, builder, op);
              setAPIifType<EwLeOp>(cfg.api, builder, op);
              setAPIifType<EwGtOp>(cfg.api, builder, op);
              setAPIifType<EwGeOp>(cfg.api, builder, op);
          
          }
        );
    }
}

std::unique_ptr<Pass> daphne::createKernelSelectionPass(const DaphneUserConfig& cfg)
{
    return std::make_unique<KernelSelectionPass>(cfg);
}
