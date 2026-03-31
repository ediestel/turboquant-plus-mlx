// Fused rotate + quantize Metal kernel.
//
// Single GPU dispatch: apply rotation matrix → normalize → nearest centroid lookup → store.
// Eliminates intermediate memory round-trips vs chaining separate MLX ops.
//
// Buffer layout:
//   0: input      — float*, shape (batch, d) — vectors to compress
//   1: rotation   — float*, shape (d, d)     — Haar rotation matrix (row-major)
//   2: centroids  — float*, shape (n_c,)     — sorted codebook centroids
//   3: boundaries — float*, shape (n_c - 1,) — midpoints between centroids
//   4: indices    — uint8*, shape (batch, d)  — output: nearest centroid per coord
//   5: norms      — float*, shape (batch,)    — output: ||x||_2 per vector
//   constants:
//     d           — vector dimension
//     n_c         — number of centroids (2^bit_width)
//
// Dispatch: one thread per (batch_idx, coord_idx) pair — grid (batch, d)

#include <metal_stdlib>
using namespace metal;

// Binary search on sorted boundaries array — O(log n_c)
inline uint searchsorted(device const float* boundaries, uint n_c_minus1, float val) {
    uint lo = 0, hi = n_c_minus1;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (boundaries[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

kernel void fused_rotate_quantize(
    device const float* input      [[buffer(0)]],
    device const float* rotation   [[buffer(1)]],
    device const float* centroids  [[buffer(2)]],
    device const float* boundaries [[buffer(3)]],
    device uint8_t*     indices    [[buffer(4)]],
    device float*       norms      [[buffer(5)]],
    constant uint&      d          [[buffer(6)]],
    constant uint&      n_c        [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]   // (batch_idx, coord_idx)
)
{
    uint batch_idx = gid.x;
    uint coord_idx = gid.y;

    if (coord_idx >= d) return;

    uint in_base = batch_idx * d;

    // --- Step 1: Compute ||x||_2 (one thread per vector — coord 0 does it) ---
    // For simplicity, all threads in the batch row compute norm in parallel via
    // atomic reduction. Each thread contributes x[i]^2.
    // NOTE: norm is written by coord 0 after a barrier; other threads read from
    // the norms buffer. In practice, dispatch a separate norm pass or use
    // threadgroup reduction. This kernel assumes norms[] is pre-filled by the host.
    // (See mlx/kv_cache.py which pre-computes norms via mx.linalg.norm.)

    // --- Step 2: Normalize and rotate ---
    float norm = norms[batch_idx];
    float safe_norm = (norm > 0.0f) ? norm : 1.0f;

    // y[coord_idx] = rotation[coord_idx, :] · (x / norm)
    float y = 0.0f;
    uint rot_base = coord_idx * d;
    for (uint j = 0; j < d; ++j) {
        y += rotation[rot_base + j] * (input[in_base + j] / safe_norm);
    }

    // --- Step 3: Nearest centroid via binary search on boundaries ---
    uint idx = searchsorted(boundaries, n_c - 1, y);

    // --- Step 4: Store ---
    indices[batch_idx * d + coord_idx] = (uint8_t)idx;
}
