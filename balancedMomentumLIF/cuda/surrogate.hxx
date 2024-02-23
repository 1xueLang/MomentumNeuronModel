#pragma once

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.1415926f
#endif

// LeakyReLU
#define __SURO_FUNC_1(x, alpha) sg = 1.0;

// sigmoid
#define __SURO_FUNC_2(x, alpha)                                  \
    sg = 1.0 / (1.0 + expf(-alpha * x));                         \
    sg = (1.0 - sg) * sg * alpha;
// arctan
#define __SURO_FUNC_3(x, alpha)                                  \
    sg = alpha / 2 / (1 + powf(CUDART_PI_F / 2 * alpha * x, 2));

// zif
#define __SURO_FUNC_4(x, alpha)                                  \
    sg = (abs(x) <= (1.0f / alpha)) * (alpha - powf(alpha, 2) * abs(x));

#define __SWITCH_SG__(suro, over_th)                             \
    switch(suro) {                                               \
    case 1:                                                      \
    __SURO_FUNC_1(over_th, alpha) break;                         \
    case 2:                                                      \
    __SURO_FUNC_2(over_th, alpha) break;                         \
    case 3:                                                      \
    __SURO_FUNC_3(over_th, alpha) break;                         \
    case 4:                                                      \
    __SURO_FUNC_4(over_th, alpha) break;                         \
    }