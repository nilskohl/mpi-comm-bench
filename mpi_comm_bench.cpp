/// A simple MPI ring bandwidth benchmark with optional CUDA support.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <optional>
#include <thread>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
static void checkCuda( cudaError_t err, const char* msg )
{
    if ( err != cudaSuccess )
    {
        fprintf( stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString( err ) );
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
}
#endif

// Returns true if the flag exists (e.g., --gpu)
bool has_flag( int argc, char** argv, const std::string& flag )
{
    for ( int i = 1; i < argc; ++i )
    {
        if ( argv[i] == flag )
        {
            return true;
        }
    }
    return false;
}

// Returns the value for flags like:
//   --msg 1024
//   --msg=1024
// Returns std::nullopt if not present.
std::optional< std::string > get_flag_value( int argc, char** argv, const std::string& flag )
{
    std::string prefix = flag + "=";
    for ( int i = 1; i < argc; ++i )
    {
        std::string arg = argv[i];
        if ( arg.rfind( prefix, 0 ) == 0 )
        {
            return arg.substr( prefix.size() ); // "--flag=value"
        }
        if ( arg == flag && i + 1 < argc )
        {
            return argv[i + 1]; // "--flag value"
        }
    }
    return std::nullopt;
}

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    int rank, num_processes;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &num_processes );

    size_t msg_size     = 1024 * 1024;
    double interval_sec = 1.0;

    auto msg_size_opt = get_flag_value( argc, argv, "--msg-size" );
    if ( msg_size_opt.has_value() )
    {
        msg_size = static_cast< size_t >( std::stod( msg_size_opt.value() ) );
    }

    const bool use_gpu = has_flag( argc, argv, "--gpu" );

    if ( rank == 0 )
    {
        std::cout << "Ring comm benchmark." << std::endl;
        std::cout << "Message size: " << msg_size << " bytes (~" << static_cast< double >( msg_size ) / 1e9 << " GB)."
                  << std::endl;
        std::cout << "Interval:     " << interval_sec << " seconds." << std::endl;
        std::cout << "GPU mode:     " << ( use_gpu ? "on" : "off" ) << std::endl;
    }

    int next = ( rank + 1 ) % num_processes;
    int prev = ( rank - 1 + num_processes ) % num_processes;

    unsigned char* send_buf = nullptr;
    unsigned char* recv_buf = nullptr;

    if ( use_gpu )
    {
#ifdef USE_CUDA
        checkCuda( cudaMalloc( &send_buf, msg_size ), "cudaMalloc" );
        checkCuda( cudaMalloc( &recv_buf, msg_size ), "cudaMalloc" );

        checkCuda( cudaMemset( send_buf, 0, msg_size ), "cudaMemset" );
        checkCuda( cudaMemset( recv_buf, 0, msg_size ), "cudaMemset" );
#else
        if ( rank == 0 )
            fprintf( stderr, "GPU mode requested but binary not built with USE_CUDA=1.\n" );
        MPI_Abort( MPI_COMM_WORLD, 1 );
#endif
    }
    else
    {
        send_buf = static_cast< unsigned char* >( malloc( msg_size ) );
        recv_buf = static_cast< unsigned char* >( malloc( msg_size ) );
    }

    while ( true )
    {
        if ( interval_sec > 0 )
        {
            std::this_thread::sleep_for( std::chrono::duration< double >( interval_sec ) );
        }

        MPI_Barrier( MPI_COMM_WORLD );

        auto t0 = std::chrono::steady_clock::now();
        MPI_Sendrecv(
            send_buf,
            static_cast< int >( msg_size ),
            MPI_UNSIGNED_CHAR,
            next,
            0,
            recv_buf,
            static_cast< int >( msg_size ),
            MPI_UNSIGNED_CHAR,
            prev,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE );
        auto t1 = std::chrono::steady_clock::now();

        const double dt = std::chrono::duration< double >( t1 - t0 ).count();
        const double bw = static_cast< double >( 2 * msg_size ) / dt / 1e9; // GB/s

        // One sample only: local stats = the single sample
        double local_min_bw = bw;
        double local_max_bw = bw;
        double local_avg_bw = bw;

        double global_min_bw, global_max_bw, global_sum_bw;
        MPI_Reduce( &local_min_bw, &global_min_bw, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD );
        MPI_Reduce( &local_max_bw, &global_max_bw, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
        MPI_Reduce( &local_avg_bw, &global_sum_bw, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );

        double global_avg_bw = global_sum_bw / num_processes;

        // One sample only: local stats = the single sample
        double local_min_dt = dt;
        double local_max_dt = dt;
        double local_avg_dt = dt;

        double global_min_dt, global_max_dt, global_sum_dt;
        MPI_Reduce( &local_min_dt, &global_min_dt, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD );
        MPI_Reduce( &local_max_dt, &global_max_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
        MPI_Reduce( &local_avg_dt, &global_sum_dt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );

        double global_avg_dt = global_sum_dt / num_processes;

        if ( rank == 0 )
        {
            std::cout << std::fixed << std::setprecision( 3 ) << "Bandwidth (send + recv): min = " << std::setw( 10 )
                      << global_min_bw << " GB/s | max = " << std::setw( 10 ) << global_max_bw
                      << " GB/s | avg = " << std::setw( 10 ) << global_avg_bw
                      << " GB/s || Duration (send + recv): min = " << std::setw( 10 ) << global_min_dt * 1e3
                      << " ms | max = " << std::setw( 10 ) << global_max_dt * 1e3 << " ms | avg = " << std::setw( 10 )
                      << global_avg_dt * 1e3 << " ms" << std::endl;
        }
    }

    if ( use_gpu )
    {
#ifdef USE_CUDA
        cudaFree( send_buf );
        cudaFree( recv_buf );
#endif
    }
    else
    {
        free( send_buf );
        free( recv_buf );
    }

    MPI_Finalize();

    return 0;
}