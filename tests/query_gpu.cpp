#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Query device 0

    std::cout << "Device name: " << prop.name << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Number of multiprocessors: " << prop.multiProcessorCount << "\n";

    int totalThreads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    std::cout << "Estimated max concurrent threads: " << totalThreads << "\n";

    return 0;
}
