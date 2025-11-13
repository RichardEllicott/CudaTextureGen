/*

custom Cuda objects, designed to automaticly allocate and free memory, download etc

*/

#pragma once

#include "core/cuda/cuda_array_2d.cuh" // stores a local side array that can allocate, upload and download easy
#include "core/cuda/cuda_array_3d.cuh" // stores a local side array that can allocate, upload and download easy
#include "core/cuda/curand_array_2d.cuh" // manages random states for 2D
#include "core/cuda/device_array.cuh" // device side only array allocation with some download/upload features, unused atm
#include "core/cuda/stream.cuh" // abstracts a stream
#include "core/cuda/struct.cuh" // uploads simple structs to memory like pars
