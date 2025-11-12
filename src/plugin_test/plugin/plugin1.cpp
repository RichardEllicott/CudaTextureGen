#include <string>
#include "tests.cuh" // 🚧


extern "C" {

// Exported function
#ifdef _WIN32
__declspec(dllexport)
#endif
const char* plugin_entry() {

    // tests::cuda_hello();
    tests::print_debug_info();

    static std::string msg = "Hello from minimal plugin!";
    return msg.c_str();
}

}
