/*

typecasters convert C++ types to Python types automaticly (instead of registering an object as a bing which creates a Python object)

*/
#pragma once

#include "typecasters/cuda_vector.h" // for float2, float3, float4, int2, int3....
#include "typecasters/ref.h"         // for core::Ref
#include "typecasters/cuda_array.h" // core::cuda::types::array