#ifndef SRC_IR_DAPHNEIR_DAPHNEPUSHDOWNTRAITS_H
#define SRC_IR_DAPHNEIR_DAPHNEPUSHDOWNTRAITS_H

namespace mlir::OpTrait {

// ============================================================================
// Traits determining advanced push-down rewrite possibilities
// ============================================================================

/**
 * @brief This trait is for operations that are linear and might allow for push-downs into functions that
 * act on e.g. ranges
 */
template <class ConcreteOp> class PushDownLinear : public TraitBase<ConcreteOp, PushDownLinear> {};

/**
 * @brief This trait is for operations that are using an increment value which
 * needs to be accounted for during push-down optimizations
 */
template <class ConcreteOp> class PushDownIncrementUpdate : public TraitBase<ConcreteOp, PushDownIncrementUpdate> {};

} // namespace mlir::OpTrait

#endif // SRC_IR_DAPHNEIR_DAPHNEPUSHDOWNTRAITS_H
