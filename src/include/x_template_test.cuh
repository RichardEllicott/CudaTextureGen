/*

testing a special "x_template"

this template will invoke both the header and binding headers with automatic memory management templates etc
the only thing that needs to be written is the .cu file

the .cu file MUST implement TEMPLATE_CLASS_NAME::process()


⚠️ WARNING: this pattern can give rise to cryptic errors

*/

// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME XTemplateTest
#define TEMPLATE_NAMESPACE x_template_test

#define TEMPLATE_CLASS_PARAMETERS \
    X(size_t, width, 256)         \
    X(size_t, height, 256)        \
    X(size_t, _block, 16)         \
    X(float, test, 0.0)

#define TEMPLATE_CLASS_MAPS \
    X(float, image)
// ════════════════════════════════════════════════ //

#include "x_template/x_template.cuh" // including here should create the entire header
