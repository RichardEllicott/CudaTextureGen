/*

undefine the definitions to avoid macro pollution

included at the bottom of the bind headers that use the template pattern

*/
#undef TEMPLATE_CLASS_NAME             // class name
#undef TEMPLATE_NAMESPACE              // namespace
#undef TEMPLATE_CLASS_PARAMETERS       // pars for structure
#undef TEMPLATE_CLASS_MAPS             // 2d maps with host side copy
#undef TEMPLATE_CLASS_DEVICE_ARRAYS    // 1d private device arrays
#undef TEMPLATE_CLASS_TYPES            // enumerators
#undef TEMPLATE_CLASS_DEBUG_DATA       // used to record debug data
#undef TEMPLATE_DEBUG_OUTPUTS          // used to record debug data
#undef TEMPLATE_CLASS_DEVICE_ARRAY_1DS // device side only 1d array
#undef TEMPLATE_CLASS_DEVICE_ARRAY_2DS // device side only 2d array
#undef TEMPLATE_CLASS_DEVICE_ARRAY_3DS // device side only 3d array
#undef TEMPLATE_CLASS_DEVICE_ARRAY_NS  // device side only ND arrays
#undef TEMPLATE_CLASS_DEVICE_ARRAY_NDS // device side only ND arrays

#undef STRINGIFY
#undef EXPAND_AND_STRINGIFY

// unused most likely
#undef TEMPLATE_CLASS_METHODS

// layers in Erosion8.cuh
#undef LAYER_NAME_DEFAULT
#undef LAYER_RESISTANCE_DEFAULT
#undef LAYER_YIELD_DEFAULT
#undef LAYER_PERMEABILITY_DEFAULT
#undef LAYER_THRESHOLD_DEFAULT
