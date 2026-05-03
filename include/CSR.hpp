#ifndef CSR_HPP
#define CSR_HPP

#include "common.h"
#include "utils.cuh"


namespace spmm {

/* Data type, index type */
template <typename D, typename I>
class CSR
{
    using COOTuple = std::tuple<I, I, D>;

public:

    CSR(size_t _m, size_t _n, size_t _nnz,
        D * _d_vals, I * _d_colinds, 
        I * _d_rowptrs):
        m(_m), n(_n), nnz(_nnz),
        d_vals(_d_vals), d_colinds(_d_colinds),
        d_rowptrs(_d_rowptrs)
        {}


    CSR(CSR& other):
        m(other.m), n(other.n), nnz(other.nnz),
        d_vals(other.d_vals), d_colinds(other.d_colinds),
        d_rowptrs(other.d_rowptrs)
    {}


    CSR(std::string& path, cusparseHandle_t& handle)
    {
        read_mm(path, handle);
    }


    void dump(std::ofstream& ofs)
    {
        ofs<<"VALS:"<<std::endl;
        utils::write_device_buffer(nnz, d_vals, ofs);
        ofs<<"COLINDS:"<<std::endl;
        utils::write_device_buffer(nnz, d_colinds, ofs);
        ofs<<"ROWPTRS:"<<std::endl;
        utils::write_device_buffer(m+1, d_rowptrs, ofs);
    }


    cusparseSpMatDescr_t to_cusparse_spmat()
    {
        cusparseSpMatDescr_t A;
        CUSPARSE_CHECK(cusparseCreateCsr(&A, 
                                         m, n, nnz,
                                         d_rowptrs, d_colinds, d_vals,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_32F));
        return A;
    }


    void abs()
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        utils::abs_transform(d_vals, nnz);
        CUDA_CHECK(cudaDeviceSynchronize());
    }


    void destroy()
    {
        CUDA_FREE_SAFE(d_vals);
        CUDA_FREE_SAFE(d_colinds);
        CUDA_FREE_SAFE(d_rowptrs);
    }


    inline size_t get_rows() const { return m; }
    inline size_t get_cols() const { return n; }
    inline size_t get_nnz() const { return nnz; }

    inline size_t get_val_bytes() const { return sizeof(D); }
    inline size_t get_ind_bytes() const { return sizeof(I); }
    inline size_t get_tuple_bytes() const { return sizeof(I)*2 + sizeof(D); }

    inline D * get_vals() const { return d_vals; }
    inline I * get_colinds() const { return d_colinds; }
    inline I * get_rowptrs() const { return d_rowptrs; }

private:


    void read_mm(std::string& path, cusparseHandle_t& handle)
    {
        std::ifstream infile;
        infile.open(path);

        std::string line;
        bool first = true;
        bool symmetric = false;
        bool pattern = false;
        while (std::getline(infile, line))
        {

            // Is this symmetric?
            if (first)
            {
                if (line.find("symmetric") != std::string::npos)
                {
                    symmetric = true;
                    std::cout<<"Matrix is symmetric"<<std::endl;
                }
                if (line.find("pattern") != std::string::npos)
                {
                    pattern = true;
                    std::cout<<"Matrix is pattern"<<std::endl;
                }
                first = false;
            }


            if (line.find('%')!=std::string::npos) continue; // still in header

            // Out of header, read in rows, cols, nnz
            std::istringstream iss(line);
            iss>>this->m>>this->n>>this->nnz;
            break;
        }



        std::vector<COOTuple> tuples;

        while (std::getline(infile, line))
        {
            std::istringstream iss(line);
            
            I rid, cid;
            D val;
            if (pattern)
            {
                iss>>rid>>cid;
                val = D(1.0);
            }
            else
            {
                iss>>rid>>cid>>val;
            }

            COOTuple t {rid-1, cid-1, val};
            tuples.push_back(t);

            if (symmetric && rid!=cid)
            {
                tuples.push_back({cid-1, rid-1, val});
            }
        }

        this->nnz = tuples.size();

        std::cout << "ROWS: " << this->m << std::endl 
                  << "COLS: " << this->n << std::endl 
                  << "NNZ: " << this->nnz << std::endl; 


        auto row_comp = [](COOTuple& t1, COOTuple& t2)
        {
            if (std::get<0>(t1) == std::get<0>(t2))
                return std::get<1>(t1) < std::get<1>(t2);
            return std::get<0>(t1) < std::get<0>(t2);
        };

        std::sort(tuples.begin(), tuples.end(), row_comp);

        std::vector<D> h_vals;
        std::vector<I> h_colinds;
        std::vector<I> h_rowinds;

        std::for_each(tuples.begin(), tuples.end(),
            [&](auto& t)mutable
            {
                h_rowinds.push_back(std::get<0>(t));
                h_colinds.push_back(std::get<1>(t));
                h_vals.push_back(std::get<2>(t));
            }
        );

        I * d_rowinds;

        CUDA_CHECK(cudaMalloc(&d_vals, sizeof(D)*nnz));
        CUDA_CHECK(cudaMalloc(&d_colinds, sizeof(I)*nnz));
        CUDA_CHECK(cudaMalloc(&d_rowinds, sizeof(I)*nnz));

        CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), sizeof(D)*nnz, 
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colinds, h_colinds.data(), sizeof(I)*nnz, 
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rowinds, h_rowinds.data(), sizeof(I)*(nnz), 
                                cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_rowptrs, sizeof(I)*(m+1)));
        CUSPARSE_CHECK(cusparseXcoo2csr(handle,
                                        (int*)d_rowinds,
                                        nnz,
                                        m,
                                        (int*)d_rowptrs,
                                        CUSPARSE_INDEX_BASE_ZERO));
        CUDA_CHECK(cudaFree(d_rowinds));

    }

    size_t m, n, nnz;
    D * d_vals;
    I * d_colinds, * d_rowptrs;

};

using csr_t = CSR<float, uint32_t>;

}


#endif
