#pragma once

#include "compiler/utils/DaphneTypeConverter.h"
#include <mlir/IR/PatternMatch.h>

using namespace mlir;

void populateDaphneToLinalgPatterns(DaphneTypeConverter &converter, RewritePatternSet &patterns);
