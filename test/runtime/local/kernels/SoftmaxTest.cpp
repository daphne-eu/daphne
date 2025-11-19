#include "run_tests.h"

#include "runtime/local/datagen/GenGivenVals.h"
#include "runtime/local/kernels/Softmax.h"

template <class DT> void check(const DT *in, const DenseMatrix<typename DT::VT> *exp, DaphneContext *dctx) {
    DenseMatrix<typename DT::VT> *res = nullptr;
    Softmax<DenseMatrix<typename DT::VT>, DT>::apply(res, in, dctx);
    CHECK(Approx(*(res->getValues())).epsilon(1e-6) == *(exp->getValues()));
}

TEMPLATE_PRODUCT_TEST_CASE("NN::Softmax", TAG_DNN, (DenseMatrix, CSRMatrix), (float, double)) {
    auto dctx = setupContextAndLogger();
    using DT = TestType;
    using VT = typename DT::VT;
    DT *input = nullptr;
    DenseMatrix<VT> *result = nullptr;

    if (std::is_same<DT, DenseMatrix<typename DT::VT>>::value) {
        input = genGivenVals<DT>(3, {-3, -2, -1, 0, 1, 2, 3, 4, 5});
        result = genGivenVals<DenseMatrix<VT>>(3, {0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
                                                   0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
                                                   0.09003057317038046, 0.24472847105479764, 0.6652409557748218});
        check(input, result, dctx.get());
    } else {
        input = genGivenVals<DT>(3, {0, 0, 1, 0, 53, 4, 0, 0, 5});
        result = genGivenVals<DenseMatrix<VT>>(3, {0.21194155761708544, 0.21194155761708544, 0.5761168847658291,
                                                   9.602680054508676e-24, 1.0, 5.242885663363464e-22,
                                                   0.006648354478866004, 0.006648354478866004, 0.986703291042268});
        check(input, result, dctx.get());
    }
    DataObjectFactory::destroy(input);
    DataObjectFactory::destroy(result);
}
