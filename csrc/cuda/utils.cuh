#pragma once

#include "../extensions.h"

#define CHECK_CUDA(x) PD_CHECK(x.is_gpu(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) PD_CHECK(x, "Input mismatch")

#define PD_DISPATCH_DEGREE_TYPES(degree, ...)     \
  [&] {                                           \
    switch (degree) {                             \
      case 1: {                                   \
        const int64_t DEGREE = 1;                 \
        return __VA_ARGS__();                     \
      }                                           \
      case 2: {                                   \
        const int64_t DEGREE = 2;                 \
        return __VA_ARGS__();                     \
      }                                           \
      case 3: {                                   \
        const int64_t DEGREE = 3;                 \
        return __VA_ARGS__();                     \
      }                                           \
      default:                                    \
        PD_THROW("Basis degree not implemented"); \
    }                                             \
  }()
