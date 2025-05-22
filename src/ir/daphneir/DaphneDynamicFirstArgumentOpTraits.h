
#ifndef SRC_IR_DAPHNEIR_DAPHNEDYNAMICFIRSTARGUMENTOPTRAITS_H
#define SRC_IR_DAPHNEIR_DAPHNEDYNAMICFIRSTARGUMENTOPTRAITS_H

namespace mlir::OpTrait {

// ============================================================================
// Traits determining pushdown rewrite possibilities
// ============================================================================

// Dynamic First Argument

template <class ConcreteOp> class DynamicFirstArgument : public TraitBase<ConcreteOp, DynamicFirstArgument> {};

} // namespace mlir::OpTrait

#endif // SRC_IR_DAPHNEIR_DAPHNEDYNAMICFIRSTARGUMENTOPTRAITS_H
