# Notes

## Branching Costs in CUDA

In CUDA, branching (especially divergent branches within a warp) can be costly

```cpp
float signed_pow(float x, float r) {
    return (x >= 0) ? pow(x, r) : -pow(-x, r);
}
```

You can rewrite the signed power function using branchless arithmetic, which is often faster and more warp-friendly:

```cpp
__device__ float signed_pow(float x, float r) {
    float s = copysignf(1.0f, x);       // +1 for x ≥ 0, -1 for x < 0
    return s * powf(fabsf(x), r);
}
```

```cpp
__device__ double signed_pow(double x, double r) {
    double s = copysign(1.0, x);
    return s * pow(fabs(x), r);
}
```

posmod examples (no a control-flow branch, but a data-level conditional)

```cpp
__device__ int posmod(int x, int m) {
    int r = x % m;
    return r + ((r < 0) * m);
}

__device__ float posmodf(float x, float m) {
    float r = fmodf(x, m);
    return r + ((r < 0.0f) * m);
}
```

## Simple CPU thread Async

```cpp
auto future = std::async(std::launch::async, [&]() {
    erosion_object.launch_kernel();
});

// Do other work...

future.get();  // Wait for completion
```
