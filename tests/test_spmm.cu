
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusparse.h>

#include <cub/cub.cuh>

#include "spmm.cuh"
#include "CSR.hpp"
#include "test_common.h"


using namespace testing;
using namespace spmm;


/*
 * fl(AX) = Y + deltaY
 * |deltaY| <= u * n * |A||X|
 * h_Y_abs = |A||X| = |A|X
 * |deltaY[i,j]| <= u * n * h_Y_abs[i,j]
 */
void spmm_ewise_comp(float * h_Y_computed, float * h_Y_correct, 
                     float * h_Y_abs, uint32_t * nnz_per_row,
                     const size_t m, const size_t n, const size_t k)
{
    for (size_t i=0; i<m; i++)
    {
        for (size_t j=0; j<k; j++)
        {
            size_t idx = j + i * k;
            float bound = FLT_EPSILON * 2 * nnz_per_row[i] * h_Y_abs[idx];
            if ( fabs(h_Y_computed[idx] - h_Y_correct[idx]) > bound )
            {
                std::cerr<<RED<<"Expected "<<h_Y_correct[idx]<<", got "<<h_Y_computed[idx]<<RESET<<std::endl;
                exit(1);
            }
        }
    }
}


void check_correctness(csr_t & csr_A, 
                       float * d_X, float * d_Y, 
                       cusparseSpMatDescr_t A, cusparseDnMatDescr_t X, cusparseDnMatDescr_t Y,
                       const size_t k)
{
    cusparseHandle_t cusparseHandle;
    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

    const float alpha = 1.0;
    const float beta = 0.0;

    auto m = csr_A.get_rows();
    auto n = csr_A.get_cols();
    auto nnz = csr_A.get_nnz();

    size_t buf_size;
    void * buf;

    // Correct SpMM
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, A, X,
                                           &beta, Y,
                                           CUDA_R_32F,
                                           CUSPARSE_SPMM_CSR_ALG3,
                                           &buf_size));
    CUDA_CHECK(cudaMalloc(&buf, buf_size));
    CUSPARSE_CHECK(cusparseSpMM(cusparseHandle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, A, X,
                                &beta, Y,
                                CUDA_R_32F,
                                CUSPARSE_SPMM_CSR_ALG3,
                                buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    float * h_correct = new float[m*k];
    CUDA_CHECK(cudaMemcpy(h_correct, d_Y, sizeof(float)*m*k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_Y, 0, sizeof(float)*m*k));


    // Student SpMM implementation
    SpMM_wrapper(csr_A, d_X, d_Y, k);
    float * h_computed = new float[m*k];
    CUDA_CHECK(cudaMemcpy(h_computed, d_Y, sizeof(float)*m*k, cudaMemcpyDeviceToHost));


    // Correct SpMM with |A| -- used for componentwise error bound check
    // It should be fine if we modify A at this point
    csr_A.abs();
    CUSPARSE_CHECK(cusparseSpMM(cusparseHandle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, A, X,
                                &beta, Y,
                                CUDA_R_32F,
                                CUSPARSE_SPMM_CSR_ALG3,
                                buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    float * h_abs = new float[m*k];
    CUDA_CHECK(cudaMemcpy(h_abs, d_Y, sizeof(float)*m*k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_Y, 0, sizeof(float)*m*k));

    uint32_t * nnz_per_row = new uint32_t[m];
    uint32_t * h_rowptrs = new uint32_t[m+1];
    CUDA_CHECK(cudaMemcpy(h_rowptrs, csr_A.get_rowptrs(), sizeof(uint32_t) * (m + 1), cudaMemcpyDeviceToHost));
    for (int i=0; i<m; i++)
    {
        nnz_per_row[i] = h_rowptrs[i+1] - h_rowptrs[i];
    }

    spmm_ewise_comp(h_computed, h_correct, h_abs, nnz_per_row, m, n, k);

    std::cout<<GREEN<<"Correctness for SpMM passed!"<<RESET<<std::endl;


    CUDA_CHECK(cudaFree(buf));
    CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));

    delete[] h_correct;
    delete[] h_computed;
    delete[] h_abs;
    delete[] nnz_per_row;
    delete[] h_rowptrs;
}


int main(int argc, char ** argv)
{
    cusparseHandle_t cusparseHandle;
    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));
    const size_t n_iters = 100;

    if (argc != 3)
    {
        std::cerr<<"Usage: ./test_spmm <path/to/mat> <k>"<<std::endl;
        exit(1);
    }

    std::string matname = std::string(argv[1]);
    size_t k = std::atol(argv[2]);


    std::cout<<BLUE<<"===== Running SpMM ("<<matname<<", k = "<<k<<") ====="<<RESET<<std::endl;


    /* IO */
    std::cout<<BLUE<<"Reading in "<<matname<<RESET<<std::endl;
    csr_t csr_A(matname, cusparseHandle);
    std::cout<<BLUE<<"Done!"<<RESET<<std::endl;


    /* Initialization */
    auto A = csr_A.to_cusparse_spmat();
    auto m = csr_A.get_rows();
    auto n = csr_A.get_cols();
    auto nnz = csr_A.get_nnz();

    float * d_X;
    init_dense_mat(n, k, &d_X);

    float * d_Y;
    CUDA_CHECK(cudaMalloc(&d_Y, sizeof(float)*m*k));
    CUDA_CHECK(cudaMemset(d_Y, 0, sizeof(float)*m*k));

    auto X = cusparse_dense_mat(d_X, n, k);
    auto Y = cusparse_dense_mat(d_Y, m, k);


    /* Correctness check */
    std::cout<<BLUE<<"Running correctness check"<<RESET<<std::endl;
    check_correctness(csr_A, d_X, d_Y, A, X, Y, k);
    

    /* Benchmark cuSPARSE */
    std::cout<<BLUE<<"Running cusparse benchmark"<<RESET<<std::endl;
    const float alpha = 1.0;
    const float beta = 0.0;
    const char * label_cusparse = "SpMM_cusparse";

    start_timer(label_cusparse);

    void * buf = nullptr;
    size_t buf_size = 0;
    for (int i=0; i<n_iters; i++) 
    {
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparseHandle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, A, X,
                                               &beta, Y,
                                               CUDA_R_32F,
                                               CUSPARSE_SPMM_CSR_ALG2,
                                               &buf_size));
        CUDA_CHECK(cudaMalloc(&buf, buf_size));
        CUSPARSE_CHECK(cusparseSpMM(cusparseHandle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, A, X,
                                    &beta, Y,
                                    CUDA_R_32F,
                                    CUSPARSE_SPMM_CSR_ALG2,
                                    buf));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(buf));
    }

    end_timer(label_cusparse);
    measure_gflops(label_cusparse, 2*nnz*k*n_iters);
    print_time(label_cusparse);
    print_gflops(label_cusparse);


    /* Benchmark student implementation */
    std::cout<<BLUE<<"Running student benchmark"<<RESET<<std::endl;
    const char * label_spmm_student = "SpMM_student";

    start_timer(label_spmm_student);

    for (int i=0; i<n_iters; i++) 
    {
        SpMM_wrapper(csr_A, d_X, d_Y, k);
    }

    end_timer(label_spmm_student);
    measure_gflops(label_spmm_student, 2*nnz*k*n_iters);
    print_time(label_spmm_student);
    print_gflops(label_spmm_student);


    /* Cleanup */
    CUSPARSE_CHECK(cusparseDestroySpMat(A));
    CUSPARSE_CHECK(cusparseDestroyDnMat(X));
    CUSPARSE_CHECK(cusparseDestroyDnMat(Y));

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));

    csr_A.destroy();

    CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));


    return 0;
}
