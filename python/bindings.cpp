#include <nanobind/nanobind.h>
namespace nb = nanobind;

NB_MODULE(cuda_hello, m) {
    m.def("hello", []() {
        return "Hello from Python!";
    });
}


