/*

custom Cuda objects, designed to automaticly allocate and free memory, download etc

*/

#pragma once

// ❌ depreciating these
#include "core/cuda/cuda_array_2d.cuh" // stores a local side array that can allocate, upload and download easy
#include "core/cuda/cuda_array_3d.cuh" // stores a local side array that can allocate, upload and download easy

#include "core/cuda/curand_array_2d.cuh" // manages random states for 2D, uses device_array
#include "core/cuda/device_array.cuh"    // NEW device array N (multidimensional template with common base class)

#include "core/cuda/device_struct.cuh" // uploads simple structs to memory like pars
#include "core/cuda/stream.cuh"        // abstracts a stream
