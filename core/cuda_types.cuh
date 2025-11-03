/*

custom Cuda objects, designed to automaticly allocate and free memory, download etc

*/

#pragma once

// #include <cmath>
// #include <vector>

#include "types.h"
#include <cuda_runtime.h> // NOTE we must have header pollution with this, it SHOULD be required
#include <stdexcept>

#include "cuda_types/cuda_array_2d.cuh"
// #include "cuda_types/cuda_stream.cuh"

#include "cuda/stream.cuh"
#include "cuda/struct.cuh"

// #include "cuda/array_2d.cuh"


#include "cuda_types/cuda_array_manager.cuh"

namespace core {

} // namespace core