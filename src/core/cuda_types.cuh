/*

custom Cuda objects, designed to automaticly allocate and free memory, download etc

*/

#pragma once

// #include <cmath>
// #include <vector>

#include "types.h"
#include <cuda_runtime.h> // NOTE we must have header pollution with this, it SHOULD be required
#include <stdexcept>

#include "cuda/cuda_array_2d.cuh"
// #include "cuda/curand_array.cuh"
#include "cuda/device_array.cuh"
#include "cuda/stream.cuh"
#include "cuda/struct.cuh"
