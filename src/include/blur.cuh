/*

simple 2 pass gaussian blur with 1d kernel

*/
#pragma once

#include <cuda_runtime.h>

namespace blur {

void blur(float *host_data, int width, int height, float sigma, bool wrap);

}