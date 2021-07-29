#ifndef FOOCONTEXT_H
#define FOOCONTEXT_H

#include <iostream> // remove

struct FooContext {
    int * bar;
    
    FooContext() {
        std::cerr << "FooContext() beg" << std::endl;
        bar = new int[123];
        std::cerr << "FooContext() end" << std::endl;
    }
    
    ~FooContext() {
        std::cerr << "~FooContext() beg" << std::endl;
        delete[] bar;
        std::cerr << "~FooContext() end" << std::endl;
    }
};

#endif /* FOOCONTEXT_H */

