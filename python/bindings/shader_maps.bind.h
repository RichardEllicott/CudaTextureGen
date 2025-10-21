/*

*/
#pragma once

#include "python_helper.h"
#include "shader_maps.cuh"

namespace nb = nanobind;

void bind_shader_maps(nb::module_ &m);
