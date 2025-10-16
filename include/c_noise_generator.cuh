/*

works now for seamless started with copilot, solved with Claude:

https://claude.ai/chat/626e97e1-b6cb-4690-9dc7-db32b31ccccf


is very simple value noise at the moment


*/
#pragma once

#define C_NOISE_GENERATOR_TYPE 2 // value, gradient, domain warped

#include "core.h"

namespace c_noise_generator {

class CNoiseGenerator {
  public:
    // float scale = 1.0 / 64.0;
    float period = 8; // Period in noise space (8 units for 8x8 repetition) (must be integer for seamless)
    int seed = 42;

    void fill(float *d_out, int width, int height);
};

} // namespace c_noise_generator