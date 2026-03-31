// Fast Walsh-Hadamard Transform — Metal compute kernel.
//
// O(d log d) butterfly with shared memory tiling for d up to 4096.
// Each threadgroup processes one row (one vector) of the batch.
//
// Buffer layout:
//   0: input  — float*, shape (batch, d)
//   1: output — float*, shape (batch, d)
//   constants:
//     d        — vector length (must be power of 2, <= 4096)
//     inv_sqrt — 1.0 / sqrt(d), precomputed on host
//
// Dispatch: threads_per_threadgroup = min(d/2, 512), threadgroups = batch

#include <metal_stdlib>
using namespace metal;

kernel void fast_hadamard_transform(
    device const float* input   [[buffer(0)]],
    device float*       output  [[buffer(1)]],
    constant uint&      d       [[buffer(2)]],
    constant float&     inv_sqrt [[buffer(3)]],
    uint2 tgid   [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]]
)
{
    // Allocate shared memory — max d = 4096 floats per threadgroup
    threadgroup float shared[4096];

    uint row = tgid.x;
    uint base = row * d;

    // Load row into shared memory — each thread loads (d / tg_size) elements
    uint elems_per_thread = d / tg_size;
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint idx = tid * elems_per_thread + i;
        shared[idx] = input[base + idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Butterfly passes: h = 1, 2, 4, ..., d/2
    for (uint h = 1; h < d; h <<= 1) {
        uint stride = h * 2;
        // Each thread handles one butterfly pair
        for (uint i = tid; i < d / 2; i += tg_size) {
            uint block = (i / h) * stride;
            uint pos   = i % h;
            uint left  = block + pos;
            uint right = left + h;
            float a = shared[left];
            float b = shared[right];
            shared[left]  = a + b;
            shared[right] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write normalized output
    for (uint i = 0; i < elems_per_thread; ++i) {
        uint idx = tid * elems_per_thread + i;
        output[base + idx] = shared[idx] * inv_sqrt;
    }
}
