#include <boost/dll/import.hpp>
#include <iostream>
#include <memory>   // for std::shared_ptr

int main() {
    try {
        // Define the function type
        using add_func = int(int, int);

        // Import the symbol from the shared library
        std::shared_ptr<add_func> add =
            boost::dll::import<add_func>(
                // library path (without extension on Linux, with .dll on Windows)
#if defined(_WIN32)
                "dll_demo.dll",
#else
                "./libdemo.so",
#endif
                "add",                // symbol name
                boost::dll::load_mode::append_decorations
            );

        std::cout << "Result: " << (*add)(2, 3) << "\n";
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
