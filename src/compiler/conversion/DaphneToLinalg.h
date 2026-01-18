#pragma once

#include "compiler/utils/DaphneTypeConverter.h"
#include <mlir/IR/PatternMatch.h>

void populateDaphneToLinalgPatterns(DaphneTypeConverter &converter, mlir::RewritePatternSet &patterns);
