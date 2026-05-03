cmake_minimum_required(VERSION 3.5)
project(SpMM VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_compile_options(-w -O3)
set(CMAKE_CUDA_ARCHITECTURES "80")

find_package(CUDAToolkit)

set(GROUP_NO 0 CACHE STRING "Your group number")
if (NOT ${GROUP_NO} STREQUAL xxx)
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "CS5220Group${GROUP_NO}_hw5")
    install(FILES include/spmm.cuh DESTINATION .)
    install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
    include(CPack)
endif ()

add_executable(test_spmm tests/test_spmm.cu)

target_include_directories(test_spmm PUBLIC include)

target_link_libraries(test_spmm PUBLIC CUDA::cusparse CUDA::cudart CUDA::cublas )

