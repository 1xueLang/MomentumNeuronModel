#include "cuda/surrogate.hxx"


///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> __global__ void cudaMomentumNeuronForwardKernel0(
    T * inputs,
    T * out1, 
    T * out2, 
    T * out3,
    T * out4,
    const T tau, 
    const T th, 
    const T mt, 
    const T lamb,
    const int N, 
    const int L, 
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index1 = index0 % D;
    index1 = (index0 - index1) * L + index1;
    T last = 0., curt = 0., mont = 0., dltu = 0., sout = 0.;
    if (index0 < N * D) {

    for (int i = 0; i < L; ++i, index1 += D) {
        out1[index1] = mont;
        curt = last / tau + inputs[index1];
        out3[index1] = curt;
        sout = curt >= th ? 1. : 0.;
        out4[index1] = sout;
        dltu = curt - last;
        out2[index1] = dltu;
        mont = mt * mont + (1. - mt) * dltu;
        last = lamb * curt + (1. - lamb) * mont;
        last = curt >= th ? 0. : last;
    }
    }
}

template<typename T> __global__ void cudaMomentumNeuronBackwardKernel0(
    T * gradin1,
    T * gradin2,
    T * gradin3,
    const T * gradout,
    const T * saved1,
    const T * saved2,
    const T * saved3,
    const T tau,
    const T th,
    const T mt, 
    const T lamb,
    const int suro,
    const float alpha,
    const int N,
    const int L,
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index = index0 * L + (D - index0 % D) * (L - 1);
    
    T du = 0., dlast = 0., dm = 0., s = 0., u = 0., mont = 0., g = 0, sg = 0.;
    T g1 = 0., g2 = 0;
    if (index0 < N * D) {

    for (int i = L - 1; i >= 0; --i, index -= D) {
        u = saved3[index];
        g = gradout[index];
        s = u >= th ? 0. : 1.;
        __SWITCH_SG__(suro, (u - th));

        dm = dm * mt + dlast * s * (1. - lamb);
        du = (g - dlast * mont) * sg + dlast * lamb * ((mont - u) * sg + s) + dm * (1. - mt);
        gradin1[index] = du;
        g1 += dlast * s * (u - mont);
        mont = saved1[index];
        g2 += dm * (mont - saved2[index]);
        dlast = du / tau + dm * (mt - 1.);
    }
        gradin2[index0] = g1; 
        gradin3[index0] = g2; 
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> __global__ void cudaMomentumNeuronForwardKernel1(
    T * inputs,
    T * out1,
    T * out3,
    T * out4,
    const T tau, 
    const T th, 
    const T mt, 
    const T lamb,
    const int N, 
    const int L, 
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index1 = index0 % D;
    index1 = (index0 - index1) * L + index1;
    T last = 0., curt = 0., mont = 0., dltu = 0., sout = 0.;
    if (index0 < N * D) {

    for (int i = 0; i < L; ++i, index1 += D) {
        out1[index1] = mont;
        curt = last / tau + inputs[index1];
        out3[index1] = curt;
        sout = curt >= th ? 1. : 0.;
        out4[index1] = sout;
        dltu = curt - last;
        mont = mt * mont + (1. - mt) * dltu;
        last = lamb * curt + (1. - lamb) * mont;
        last = curt >= th ? 0. : last;
    }
    }
}

template<typename T> __global__ void cudaMomentumNeuronBackwardKernel1(
    T * gradin1,
    T * gradin2,
    const T * gradout,
    const T * saved1,
    const T * saved3,
    const T tau,
    const T th,
    const T mt, 
    const T lamb,
    const int suro,
    const float alpha,
    const int N,
    const int L,
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index = index0 * L + (D - index0 % D) * (L - 1);
    
    T du = 0., dlast = 0., dm = 0., s = 0., u = 0., mont = 0., g = 0, sg = 0.;
    T g1 = 0.;
    if (index0 < N * D) {

    for (int i = L - 1; i >= 0; --i, index -= D) {
        u = saved3[index];
        g = gradout[index];
        s = u >= th ? 0. : 1.;
        __SWITCH_SG__(suro, (u - th));

        dm = dm * mt + dlast * s * (1. - lamb);
        du = (g - dlast * mont) * sg + dlast * lamb * ((mont - u) * sg + s) + dm * (1. - mt);
        gradin1[index] = du;
        g1 += dlast * s * (u - mont);
        mont = saved1[index];
        dlast = du / tau + dm * (mt - 1.);
    }
        gradin2[index0] = g1;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> __global__ void cudaMomentumNeuronForwardKernel2(
    T * inputs,
    T * out1, 
    T * out2, 
    T * out3,
    T * out4,
    const T tau, 
    const T th, 
    const T mt, 
    const T lamb,
    const int N, 
    const int L, 
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index1 = index0 % D;
    index1 = (index0 - index1) * L + index1;
    T last = 0., curt = 0., mont = 0., dltu = 0., sout = 0.;
    if (index0 < N * D) {

    for (int i = 0; i < L; ++i, index1 += D) {
        out1[index1] = mont;
        curt = last / tau + inputs[index1];
        out3[index1] = curt;
        sout = curt >= th ? 1. : 0.;
        out4[index1] = sout;
        dltu = curt - last;
        out2[index1] = dltu;
        mont = mt * mont + (1. - mt) * dltu;
        last = lamb * curt + (1. - lamb) * mont;
        last = curt >= th ? 0. : last;
    }
    }
}

template<typename T> __global__ void cudaMomentumNeuronBackwardKernel2(
    T * gradin1,
    T * gradin3,
    const T * gradout,
    const T * saved1,
    const T * saved2,
    const T * saved3,
    const T tau,
    const T th,
    const T mt, 
    const T lamb,
    const int suro,
    const float alpha,
    const int N,
    const int L,
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index = index0 * L + (D - index0 % D) * (L - 1);
    
    T du = 0., dlast = 0., dm = 0., s = 0., u = 0., mont = 0., g = 0, sg = 0.;
    T g2 = 0;
    if (index0 < N * D) {

    for (int i = L - 1; i >= 0; --i, index -= D) {
        u = saved3[index];
        g = gradout[index];
        s = u >= th ? 0. : 1.;
        __SWITCH_SG__(suro, (u - th));

        dm = dm * mt + dlast * s * (1. - lamb);
        du = (g - dlast * mont) * sg + dlast * lamb * ((mont - u) * sg + s) + dm * (1. - mt);
        gradin1[index] = du;
        mont = saved1[index];
        g2 += dm * (mont - saved2[index]);
        dlast = du / tau + dm * (mt - 1.);
    }
        gradin3[index0] = g2; 
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> __global__ void cudaMomentumNeuronForwardKernel3(
    T * inputs,
    T * out1,
    T * out3,
    T * out4,
    const T tau, 
    const T th, 
    const T mt, 
    const T lamb,
    const int N, 
    const int L, 
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index1 = index0 % D;
    index1 = (index0 - index1) * L + index1;
    T last = 0., curt = 0., mont = 0., dltu = 0., sout = 0.;
    if (index0 < N * D) {

    for (int i = 0; i < L; ++i, index1 += D) {
        out1[index1] = mont;
        curt = last / tau + inputs[index1];
        out3[index1] = curt;
        sout = curt >= th ? 1. : 0.;
        out4[index1] = sout;
        dltu = curt - last;
        mont = mt * mont + (1. - mt) * dltu;
        last = lamb * curt + (1. - lamb) * mont;
        last = curt >= th ? 0. : last;
    }
    }
}

template<typename T> __global__ void cudaMomentumNeuronBackwardKernel3(
    T * gradin1,
    const T * gradout,
    const T * saved1,
    const T * saved3,
    const T tau,
    const T th,
    const T mt, 
    const T lamb,
    const int suro,
    const float alpha,
    const int N,
    const int L,
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index = index0 * L + (D - index0 % D) * (L - 1);
    
    T du = 0., dlast = 0., dm = 0., s = 0., u = 0., mont = 0., g = 0, sg = 0.;
    if (index0 < N * D) {

    for (int i = L - 1; i >= 0; --i, index -= D) {
        u = saved3[index];
        g = gradout[index];
        s = u >= th ? 0. : 1.;
        __SWITCH_SG__(suro, (u - th));

        dm = dm * mt + dlast * s * (1. - lamb);
        du = (g - dlast * mont) * sg + dlast * lamb * ((mont - u) * sg + s) + dm * (1. - mt);
        gradin1[index] = du;
        mont = saved1[index];
        dlast = du / tau + dm * (mt - 1);
    }
    }
}