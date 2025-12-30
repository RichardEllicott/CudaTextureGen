/*

python helper library

functions to covert numpy arrays back and forth to std::vector and core::Array2D

*/
#pragma once

#include "nanobind_helper/array_nd.h"
#include "nanobind_helper/device_array.h"
#include "nanobind_helper/numpy.h"
#include "nanobind_helper/vector.h"
#include "nanobind_helper/bind_dynamic_properties.h" // binding python property setting to a has/get/set functions

#include "nanobind_helper/ref.h" // type caster for core::Ref

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

namespace nanobind::helper {

namespace nb = nanobind;

// get nested python lists (example)
inline nb::object get_list_of_lists(int height, int width) {

    std::vector<nb::list> rows;
    rows.reserve(height);

    for (int y = 0; y < height; ++y) {
        nb::list row;
        for (int x = 0; x < width; ++x) {
            row.append(0.0f); // or some value
        }
        rows.push_back(row);
    }

    nb::list outer;
    for (auto &r : rows)
        outer.append(r);

    return outer; // Python will see a list of lists
}

// throw a Python warning, note normal exceptions should use C++ syntax
// ⚠️ USING THIS WITH LINUX CAUSED AN ERROR... it's to do with the order.... add back later?
inline void warn(const char *message, const char *category = "UserWarning") {
    nb::module_ warnings = nb::module_::import_("warnings");
    nb::module_ builtins = nb::module_::import_("builtins");
    warnings.attr("warn")(message, builtins.attr(category));
}

} // namespace nanobind::helper