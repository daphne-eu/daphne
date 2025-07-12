#ifndef SRC_IR_DAPHNEIR_DAPHNEPUSHDOWNTRAITS_H
#define SRC_IR_DAPHNEIR_DAPHNEPUSHDOWNTRAITS_H

namespace mlir::OpTrait {

// ============================================================================
// Traits determining pushdown rewrite possibilities
// ============================================================================

// Dynamic First Argument

template <class ConcreteOp> class PushDown : public TraitBase<ConcreteOp, PushDown> {};
template <class ConcreteOp> class PushDownLinear : public TraitBase<ConcreteOp, PushDownLinear> {};
template <class ConcreteOp>
class PushDownWithIntervalUpdate : public TraitBase<ConcreteOp, PushDownWithIntervalUpdate> {};

} // namespace mlir::OpTrait

#endif // SRC_IR_DAPHNEIR_DAPHNEPUSHDOWNTRAITS_H
