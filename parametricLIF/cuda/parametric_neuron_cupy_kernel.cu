#include "cuda/surrogate.hxx"


///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> __global__ void cudaParametricNeuronForwardKernel(
    T * inputs,
    T * out3,
    T * out4,
    const T tau, 
    const T th,
    const int N, 
    const int L, 
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index1 = index0 % D;
    index1 = (index0 - index1) * L + index1;
    T curt = 0., sout = 0.;
    if (index0 < N * D) {

    for (int i = 0; i < L; ++i, index1 += D) {
        curt = curt * tau + inputs[index1];
        out3[index1] = curt;
        sout = curt >= th ? 1. : 0.;
        out4[index1] = sout;
        curt = curt >= th ? 0. : curt;
    }
    }
}

template<typename T> __global__ void cudaParametricNeuronBackwardKernel(
    T * gradin1,
    T * gradin2,
    const T * gradout,
    const T * saved3,
    const T tau,
    const T th,
    const int suro,
    const float alpha,
    const int N,
    const int L,
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index = index0 * L + (D - index0 % D) * (L - 1);
    
    T u = 0., du = 0., sg = 0., g = 0.;
    if (index0 < N * D) {

    for (int i = L - 1; i >= 0; --i, index -= D) {
        u = saved3[index];
        g += du * (u >= th ? 0. : u);
        __SWITCH_SG__(suro, (u - th));

        du = gradout[index] * sg + du * (1. - (T)(u >= th) - u * sg) * tau;
        
        gradin1[index] = du;
    }
        gradin2[index0] = g;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> __global__ void cudamParametricNeuronForwardKernel(
    T * inputs,
    T * out2,
    T * out3,
    T * out4,
    const T tau, 
    const T th, 
    const T lamb,
    const int N, 
    const int L, 
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index1 = index0 % D;
    index1 = (index0 - index1) * L + index1;
    T last = 0., curt = 0., sout = 0.;
    if (index0 < N * D) {

    for (int i = 0; i < L; ++i, index1 += D) {
        out2[index1] = last;
        curt = last * tau + inputs[index1];
        out3[index1] = curt;
        sout = curt >= th ? 1. : 0.;
        out4[index1] = sout;
        last = curt + lamb * (curt - last);
        last = curt >= th ? 0. : last;
    }
    }
}

template<typename T> __global__ void cudamParametricNeuronBackwardKernel(
    T * gradin1,
    T * gradin2,
    T * gradin3,
    const T * gradout,
    const T * saved2,
    const T * saved3,
    const T tau,
    const T th,
    const T lamb,
    const int suro,
    const float alpha,
    const int N,
    const int L,
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index = index0 * L + (D - index0 % D) * (L - 1);
    
    T du = 0., dlast = 0., dm = 0., s = 0., u = 0., last = 0., sg = 0.;
    T g1 = 0., g2 = 0.;
    if (index0 < N * D) {

    for (int i = L - 1; i >= 0; --i, index -= D) {
        u = saved3[index];
        last = saved2[index];
        s = u >= th ? 0. : 1.;
        __SWITCH_SG__(suro, (u - th));

        dm = dlast * s * lamb;
        du = gradout[index] * sg - dlast * ((u + lamb * (u - last)) * sg - s) + dm;

        gradin1[index] = du;
        g1 += dlast * s * (u - last);
        dlast = du * tau - dm;
        g2 += du * last;
    }
        gradin2[index0] = g1;
        gradin3[index0] = g2;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> __global__ void cudabmParametricNeuronForwardKernel(
    T * inputs,
    T * out2,
    T * out3,
    T * out4,
    const T tau, 
    const T th, 
    const T lamb,
    const int N, 
    const int L, 
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index1 = index0 % D;
    index1 = (index0 - index1) * L + index1;
    T last = 0., curt = 0., sout = 0.;
    if (index0 < N * D) {

    for (int i = 0; i < L; ++i, index1 += D) {
        out2[index1] = last;
        curt = last * tau + inputs[index1];
        out3[index1] = curt;
        sout = curt >= th ? 1. : 0.;
        out4[index1] = sout;
        last = curt * lamb + (1. - lamb) * (curt - last);
        last = curt >= th ? 0. : last;
    }
    }
}

template<typename T> __global__ void cudabmParametricNeuronBackwardKernel(
    T * gradin1,
    T * gradin2,
    T * gradin3,
    const T * gradout,
    const T * saved2,
    const T * saved3,
    const T tau,
    const T th,
    const T lamb,
    const int suro,
    const float alpha,
    const int N,
    const int L,
    const int D) {
    int index0 = threadIdx.x + blockDim.x * blockIdx.x;
    int index = index0 * L + (D - index0 % D) * (L - 1);
    
    T du = 0., dlast = 0., dm = 0., s = 0., u = 0., last = 0., sg = 0.;
    T g1 = 0., g2 = 0.;
    if (index0 < N * D) {

    for (int i = L - 1; i >= 0; --i, index -= D) {
        u = saved3[index];
        last = saved2[index];
        s = u >= th ? 0. : 1.;
        __SWITCH_SG__(suro, (u - th));

        dm = dlast * s * (1. - lamb);
        du = gradout[index] * sg - dlast * ((u + (lamb - 1.) * last) * sg - s * lamb) + dm;

        gradin1[index] = du;
        g1 += dlast * s * last;
        dlast = du * tau - dm;
        g2 += du * last;
    }
        gradin2[index0] = g1;
        gradin3[index0] = g2;
    }
}