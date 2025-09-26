#pragma once

extern "C" void cuda_hello();


void run_blur(float* host_data, int width, int height, float sigma, bool wrap);