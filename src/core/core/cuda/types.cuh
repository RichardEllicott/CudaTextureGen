/*

custom Cuda objects, designed to automaticly allocate and free memory, download etc

*/

#pragma once

// ❌ depreciating these
#include "cuda_array_2d.cuh" // stores a local side array that can allocate, upload and download easy
#include "cuda_array_3d.cuh" // stores a local side array that can allocate, upload and download easy

#include "curand_array_2d.cuh" // manages random states for 2D, uses device_array
#include "device_array.cuh"    // NEW device array N (multidimensional template with common base class)

#include "device_struct.cuh" // uploads simple structs to memory like pars
#include "stream.cuh"        // abstracts a stream
