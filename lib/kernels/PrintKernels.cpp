#include "datastructures/Matrix.h"

#include <iostream>

#include <cassert>

extern "C"
{

    void printInt(int x)
    {
        std::cout << x << std::endl;
    }

    void printDouble(double x)
    {
        std::cout << x << std::endl;
    }

    void printMatrix(BaseMatrix * x)
    {
        if (AbstractMatrix<double> *mat =
            dynamic_cast<AbstractMatrix<double> *> (x)) {
            std::cout << *mat;
            return;
        }
        if (AbstractMatrix<float> *mat =
            dynamic_cast<AbstractMatrix<float> *> (x)) {
            std::cout << *mat;
            return;
        }
        if (AbstractMatrix<int64_t> *mat =
            dynamic_cast<AbstractMatrix<int64_t> *> (x)) {
            std::cout << *mat;
            return;
        }
        if (AbstractMatrix<int32_t> *mat =
            dynamic_cast<AbstractMatrix<int32_t> *> (x)) {
            std::cout << *mat;
            return;
        }
        assert(false && "Matrix type not recognized");
    }
}