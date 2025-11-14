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
#undef TEMPLATE_CLASS_DEVICE_ARRAY_2DS // new device side only 2d array

#undef STRINGIFY
#undef EXPAND_AND_STRINGIFY
