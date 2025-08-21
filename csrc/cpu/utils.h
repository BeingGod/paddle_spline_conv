#pragma once

#include "../extensions.h"

#define CHECK_CPU(x) PD_CHECK(x.is_cpu(), #x " must be CPU tensor")
#define CHECK_INPUT(x) PD_CHECK(x, "Input mismatch")

#define PD_DISPATCH_DEGREE_TYPES(degree, ...)     \
  [&] {                                           \
    switch (degree) {                             \
      case 1: {                                   \
        static constexpr int64_t DEGREE = 1;      \
        return __VA_ARGS__();                     \
      }                                           \
      case 2: {                                   \
        static constexpr int64_t DEGREE = 2;      \
        return __VA_ARGS__();                     \
      }                                           \
      case 3: {                                   \
        static constexpr int64_t DEGREE = 3;      \
        return __VA_ARGS__();                     \
      }                                           \
      default:                                    \
        PD_THROW("Basis degree not implemented"); \
    }                                             \
  }()
