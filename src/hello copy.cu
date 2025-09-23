// hello.cu
#include <iostream>

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    std::cout << "Hello from CPU\n";

    // Launch kernel with 5 threads
    hello_kernel<<<1, 5>>>();
    cudaDeviceSynchronize();

    return 0;
}
