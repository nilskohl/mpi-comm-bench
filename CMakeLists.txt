cmake_minimum_required(VERSION 3.14)
project(mpi_comm_bench CXX)

find_package(MPI REQUIRED)

option(USE_CUDA "Enable CUDA support" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

add_executable(mpi_comm_bench mpi_comm_bench.cpp)
target_compile_features(mpi_comm_bench PRIVATE cxx_std_17)
target_compile_options(mpi_comm_bench PRIVATE -O3)
target_link_libraries(mpi_comm_bench PRIVATE MPI::MPI_CXX)

if(USE_CUDA)
    target_compile_definitions(mpi_comm_bench PRIVATE USE_CUDA)
    target_include_directories(mpi_comm_bench PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(mpi_comm_bench PRIVATE CUDA::cudart)
endif()