//
//  main.m
//  MetalGPGPU
//
//  Created by Rajai Kumar on 07/07/24.
//


#include <metal_stdlib>
using namespace metal;
/// This is a Metal Shading Language (MSL) function  used to perform the calculation on a GPU.
kernel void compute_Values(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
   // Compute the dot product of inA and inB
    float dotProduct = inA[index] * inB[index];

    // Apply the sigmoid function to the dot product
    result[index] = 1.0 / (1.0 + exp(-dotProduct));
}
