#ifndef SPMM_CUH
#define SPMM_CUH

#include "common.h"
#include "CSR.hpp"

namespace spmm {

// ============================================================================
// Strategy: Block-per-row-group, warps split k columns.
//
// Block dim = (32, K/32). Each warp handles a 32-column slab of k.
// All warps in a block work on the SAME rows in lockstep, so global memory
// reads of A vals/colinds are amortized: only one warp's read goes to DRAM,
// the rest hit L1.
//
// Each thread holds ONE register accumulator (vs 8 in the previous version
// for k=256), so register pressure is low and occupancy is high.
// ============================================================================

template <int K_>
__global__ __launch_bounds__(K_)
void SpMM_kernel(
    const float    * __restrict__ vals,
    const uint32_t * __restrict__ colinds,
    const uint32_t * __restrict__ rowptrs,
    const float    * __restrict__ X,
    float          * __restrict__ Y,
    const uint32_t * __restrict__ row_group_begin)
{
    const int warp  = threadIdx.y;
    const int lane  = threadIdx.x;
    const int k_col = warp * 32 + lane;     // 0 .. K_-1

    const uint32_t row0 = row_group_begin[blockIdx.x];
    const uint32_t row1 = row_group_begin[blockIdx.x + 1];

    for (uint32_t row = row0; row < row1; row++) {
        const uint32_t rs = rowptrs[row];
        const uint32_t re = rowptrs[row + 1];

        float acc = 0.0f;

        // Inner loop: walk nonzeros, multiply by X[col, k_col]
        for (uint32_t jj = rs; jj < re; jj++) {
            const float    a = __ldg(&vals[jj]);
            const uint32_t c = __ldg(&colinds[jj]);
            acc += a * __ldg(&X[(size_t)c * K_ + k_col]);
        }

        Y[(size_t)row * K_ + k_col] = acc;
    }
}

// Generic fallback for k that isn't 64 or 256
__global__ void SpMM_kernel_generic(
    const int K,
    const float    * __restrict__ vals,
    const uint32_t * __restrict__ colinds,
    const uint32_t * __restrict__ rowptrs,
    const float    * __restrict__ X,
    float          * __restrict__ Y,
    const uint32_t * __restrict__ row_group_begin)
{
    const int warp  = threadIdx.y;
    const int lane  = threadIdx.x;
    const int k_col = warp * 32 + lane;
    if (k_col >= K) return;

    const uint32_t row0 = row_group_begin[blockIdx.x];
    const uint32_t row1 = row_group_begin[blockIdx.x + 1];

    for (uint32_t row = row0; row < row1; row++) {
        const uint32_t rs = rowptrs[row];
        const uint32_t re = rowptrs[row + 1];

        float acc = 0.0f;
        for (uint32_t jj = rs; jj < re; jj++) {
            const float    a = __ldg(&vals[jj]);
            const uint32_t c = __ldg(&colinds[jj]);
            acc += a * __ldg(&X[(size_t)c * K + k_col]);
        }
        Y[(size_t)row * K + k_col] = acc;
    }
}

// ----- Cached preprocessing state -----
static uint32_t * d_row_group_begin = nullptr;
static int        cached_num_groups = 0;
static uint32_t * cached_rowptrs    = nullptr;
static size_t     cached_m          = 0;

void SpMM_wrapper(csr_t& A, float * d_X, float * d_Y, const size_t k)
{
    const uint32_t m   = A.get_rows();
    const size_t   nnz = A.get_nnz();

    // ---- Build row-group boundaries (cached after first call) ----
    if (cached_rowptrs != A.get_rowptrs() || cached_m != m) {
        if (d_row_group_begin) {
            CUDA_CHECK(cudaFree(d_row_group_begin));
            d_row_group_begin = nullptr;
        }

        std::vector<uint32_t> h_rp(m + 1);
        CUDA_CHECK(cudaMemcpy(h_rp.data(), A.get_rowptrs(),
                              sizeof(uint32_t) * (m + 1), cudaMemcpyDeviceToHost));

        // Target ~1024 nnz per block: enough work to amortize launch + barrier
        // overhead, while still creating many groups for parallelism.
        size_t target = std::max((size_t)256, nnz / (size_t)(108 * 16));

        std::vector<uint32_t> h_groups;
        h_groups.reserve(nnz / target + 2);
        h_groups.push_back(0);

        uint32_t running = 0;
        for (uint32_t r = 0; r < m; r++) {
            running += h_rp[r + 1] - h_rp[r];
            if (running >= (uint32_t)target) {
                h_groups.push_back(r + 1);
                running = 0;
            }
        }
        if (h_groups.back() != m) h_groups.push_back(m);

        cached_num_groups = (int)h_groups.size() - 1;
        CUDA_CHECK(cudaMalloc(&d_row_group_begin,
                              sizeof(uint32_t) * h_groups.size()));
        CUDA_CHECK(cudaMemcpy(d_row_group_begin, h_groups.data(),
                              sizeof(uint32_t) * h_groups.size(),
                              cudaMemcpyHostToDevice));
        cached_rowptrs = A.get_rowptrs();
        cached_m       = m;
    }

    // ---- Launch ----
    // No cudaMemset of d_Y: kernel writes Y directly (=, not +=)
    if (k == 64) {
        dim3 block(32, 2);   // 64 threads
        SpMM_kernel<64><<<cached_num_groups, block>>>(
            A.get_vals(), A.get_colinds(), A.get_rowptrs(),
            d_X, d_Y, d_row_group_begin);
    }
    else if (k == 128) {
        dim3 block(32, 4);   // 128 threads
        SpMM_kernel<128><<<cached_num_groups, block>>>(
            A.get_vals(), A.get_colinds(), A.get_rowptrs(),
            d_X, d_Y, d_row_group_begin);
    }
    else if (k == 256) {
        dim3 block(32, 8);   // 256 threads
        SpMM_kernel<256><<<cached_num_groups, block>>>(
            A.get_vals(), A.get_colinds(), A.get_rowptrs(),
            d_X, d_Y, d_row_group_begin);
    }
    else {
        int nwarps = (int)((k + 31) / 32);
        dim3 block(32, nwarps);
        SpMM_kernel_generic<<<cached_num_groups, block>>>(
            (int)k,
            A.get_vals(), A.get_colinds(), A.get_rowptrs(),
            d_X, d_Y, d_row_group_begin);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

}
#endif
