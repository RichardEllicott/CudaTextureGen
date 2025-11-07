/*




*/
#include <windows.h>
#include <iostream>

// typedef int (*add_func)(int, int);
using add_func = int(*)(int, int); // new syntax


int main() {
    HMODULE lib = LoadLibraryA("dll_demo.dll");
    if (!lib) {
        std::cerr << "Failed to load DLL\n";
        return 1;
    }

    add_func add = (add_func)GetProcAddress(lib, "add");
    if (!add) {
        std::cerr << "Failed to find symbol\n";
        FreeLibrary(lib);
        return 1;
    }

    std::cout << "Result: " << add(2, 3) << "\n";

    FreeLibrary(lib);
    return 0;
}
