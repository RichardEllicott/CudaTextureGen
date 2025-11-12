#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

int main() {
    std::string path =
#ifdef _WIN32
        "plugin.dll";
#elif __APPLE__
        "./libplugin.dylib";
#else
        "./libplugin.so";
#endif

#ifdef _WIN32
    HMODULE lib = LoadLibraryA(path.c_str());
    if (!lib) {
        std::cerr << "Failed to load plugin\n";
        return 1;
    }
    auto func = (const char *(*)())GetProcAddress(lib, "plugin_entry");
#else
    void *lib = dlopen(path.c_str(), RTLD_LAZY);
    if (!lib) {
        std::cerr << "Failed to load plugin\n";
        return 1;
    }
    auto func = (const char *(*)())dlsym(lib, "plugin_entry");
#endif

    if (!func) {
        std::cerr << "Failed to find symbol\n";
        return 1;
    }

    std::cout << "Plugin says: " << func() << "\n";

#ifdef _WIN32
    FreeLibrary(lib);
#else
    dlclose(lib);
#endif

    return 0;
}
