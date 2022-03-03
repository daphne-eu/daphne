#!/usr/bin/env python3

import ctypes

libDaphneShared = ctypes.CDLL("build/src/api/shared/libDaphneShared.so")

res = libDaphneShared.doMain()
print("res: {}".format(res))