#include <windows.h>

extern "C" __declspec(dllexport)
int add(int a, int b) {
    return a + b;
}

// Optional: DLL entry point (not strictly needed for simple exports)
BOOL APIENTRY DllMain(HMODULE hModule,
                      DWORD  ul_reason_for_call,
                      LPVOID lpReserved) {
    return TRUE;
}
