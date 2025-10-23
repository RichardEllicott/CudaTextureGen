#include "template_class.h"

namespace template_class {

// --------------------------------------------------------------------------------
// Make Get/Sets
// --------------------------------------------------------------------------------
#define X(TYPE, NAME, DEFAULT_VAL)                               \
    TYPE TemplateClass::get_##NAME() const { return pars.NAME; } \
    void TemplateClass::set_##NAME(TYPE value) { pars.NAME = value; }
TEMPLATE_CLASS_PARAMETERS
#undef X
// --------------------------------------------------------------------------------

// TemplateClass::TemplateClass() {
//         // --------------------------------------------------------------------------------
//         // Set Defaults
//         // --------------------------------------------------------------------------------
// #define X(TYPE, NAME, DEFAULT_VAL) \
//     pars.NAME = DEFAULT_VAL;
//         TEMPLATE_CLASS_PARAMETERS
// #undef X
//         // --------------------------------------------------------------------------------
//     }








} // namespace template_class