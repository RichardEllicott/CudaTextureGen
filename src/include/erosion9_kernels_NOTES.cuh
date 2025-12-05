/*

*/
#pragma once

// 2D Saint-Venant (shallow-water)
#pragma region SHALLOW_WATER

#define HD_INLINE __host__ __device__ inline
constexpr float GRAVITY = 9.81f;

struct Cell {
    float h; // depth
    float u; // vel x
    float v; // vel y
    float z; // bed elevation (terrain)
};

struct Flux3 {
    float fh;  // mass flux
    float fhu; // x-momentum flux
    float fhv; // y-momentum flux
};

HD_INLINE Flux3 flux_x(const Cell &c) {
    float hu = c.h * c.u;
    float hv = c.h * c.v;
    float p = 0.5f * GRAVITY * c.h * c.h; // hydrostatic pressure term
    return {hu, hu * c.u + p, hu * c.v};
}

HD_INLINE Flux3 flux_y(const Cell &c) {
    float hu = c.h * c.u;
    float hv = c.h * c.v;
    float p = 0.5f * GRAVITY * c.h * c.h;
    return {hv, hu * c.v, hv * c.v + p};
}

HD_INLINE float wavespeed(const Cell &c) {
    float c0 = sqrtf(GRAVITY * fmaxf(c.h, 0.0f));
    float ax = fabsf(c.u) + c0;
    float ay = fabsf(c.v) + c0;
    return fmaxf(ax, ay);
}

// Rusanov interface flux and update

HD_INLINE Flux3 rusanov_flux_x(const Cell &L, const Cell &R) {
    Flux3 fL = flux_x(L);
    Flux3 fR = flux_x(R);
    float a = fmaxf(wavespeed(L), wavespeed(R));
    return {
        0.5f * (fL.fh + fR.fh) - 0.5f * a * (R.h - L.h),
        0.5f * (fL.fhu + fR.fhu) - 0.5f * a * ((R.h * R.u) - (L.h * L.u)),
        0.5f * (fL.fhv + fR.fhv) - 0.5f * a * ((R.h * R.v) - (L.h * L.v))};
}

HD_INLINE Flux3 rusanov_flux_y(const Cell &B, const Cell &T) {
    Flux3 gB = flux_y(B);
    Flux3 gT = flux_y(T);
    float a = fmaxf(wavespeed(B), wavespeed(T));
    return {
        0.5f * (gB.fh + gT.fh) - 0.5f * a * (T.h - B.h),
        0.5f * (gB.fhu + gT.fhu) - 0.5f * a * ((T.h * T.u) - (B.h * B.u)),
        0.5f * (gB.fhv + gT.fhv) - 0.5f * a * ((T.h * T.v) - (B.h * B.v))};
}

// Cell update (finite-volume on a uniform grid with spacing dx, dy):

HD_INLINE void update_cell(
    const Flux3 &FxL, const Flux3 &FxR,
    const Flux3 &FyB, const Flux3 &FyT,
    const Cell &c, float n_rough, float dx, float dy, float dt,
    Cell &out) {
    // Divergence of fluxes
    float dh = -(FxR.fh - FxL.fh) / dx - (FyT.fh - FyB.fh) / dy;
    float dhu = -(FxR.fhu - FxL.fhu) / dx - (FyT.fhu - FyB.fhu) / dy;
    float dhv = -(FxR.fhv - FxL.fhv) / dx - (FyT.fhv - FyB.fhv) / dy;

    // Bed slope source terms: -g h ∂z/∂x, -g h ∂z/∂y (use centered diffs outside this helper)
    // Friction (Manning): tau_x, tau_y
    float speed = sqrtf(c.u * c.u + c.v * c.v);
    float h_safe = fmaxf(c.h, 1e-6f);
    float tau_x = GRAVITY * n_rough * n_rough * c.u * speed / powf(h_safe, 4.0f / 3.0f);
    float tau_y = GRAVITY * n_rough * n_rough * c.v * speed / powf(h_safe, 4.0f / 3.0f);

    // Update conserved variables
    float h_new = c.h + dt * dh;
    float hu_new = c.h * c.u + dt * (dhu - h_safe * tau_x);
    float hv_new = c.h * c.v + dt * (dhv - h_safe * tau_y);

    // Reconstruct primitive variables
    h_new = fmaxf(h_new, 0.0f);
    float inv_h = (h_new > 1e-6f) ? (1.0f / h_new) : 0.0f;
    out.h = h_new;
    out.u = hu_new * inv_h;
    out.v = hv_new * inv_h;

    // Note: add bed-slope source terms outside using ∂z/∂x, ∂z/∂y and -g h ∂z
}

// Pseudocode for a 2D kernel launch over interior cells
__global__ void shallow_water_step(const Cell *in, Cell *out,
                                   int W, int H, float dx, float dy,
                                   float dt, float n_rough) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= W - 1 || y >= H - 1)
        return;

    int idx = y * W + x;
    auto C = in[idx];
    auto L = in[idx - 1];
    auto R = in[idx + 1];
    auto B = in[idx - W];
    auto T = in[idx + W];

    // Interface fluxes (left, right, bottom, top)
    Flux3 FxL = rusanov_flux_x(L, C);
    Flux3 FxR = rusanov_flux_x(C, R);
    Flux3 FyB = rusanov_flux_y(B, C);
    Flux3 FyT = rusanov_flux_y(C, T);

    // Update cell
    Cell Cout;
    update_cell(FxL, FxR, FyB, FyT, C, n_rough, dx, dy, dt, Cout);

    // Bed slope sources (optional, outside update_cell)
    // Compute centered gradients of z:
    float dzdx = (R.z - L.z) / (2.0f * dx);
    float dzdy = (T.z - B.z) / (2.0f * dy);
    float hbar = Cout.h;

    // Apply bed slope acceleration: u' += -g * dt * ∂z/∂x, v' similarly
    Cout.u += -GRAVITY * dt * dzdx;
    Cout.v += -GRAVITY * dt * dzdy;

    out[idx] = Cout;
}

#pragma endregion
