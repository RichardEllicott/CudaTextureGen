#include <iostream>

#include "core/logging.h"

using namespace std;

int main() {

    core::logging::init_console();
    core::logging::println("🐌 Hello World!\n");

    // printf("🐌 Hello printf!\n");

    return 0;
}