/*
 */
#pragma once

#include "python_helper.h"
#include "template_class.h"

namespace nb = nanobind;

namespace template_class {

inline void bind(nb::module_ &m) {

    auto ngd = nb::class_<TemplateClass>(m, "TemplateClass")
                   .def(nb::init<>()) //

    // bind get/sets
#define X(TYPE, NAME, DEFAULT_VAL) \
    .def_prop_rw(#NAME, &TemplateClass::get_##NAME, &TemplateClass::set_##NAME)
               TEMPLATE_CLASS_PARAMETERS
#undef X

        ; //
}

} // namespace template_class