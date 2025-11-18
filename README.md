# Simple MPI communication benchmark (CPU and GPU (CUDA))

Send data around in a ring-like topology (rank 0 → rank 1, rank 1 → rank 2, ..., rank N-1 → rank 0) and measures the 
time it takes.

Runs forever (sending messages around after sleeping for a short interval) until interrupted.

Can be configured to run on CPU or GPU. If run on GPU, CUDA must be installed. Each rank will be assigned exactly one 
GPU. If the number of ranks and GPUs mismatches, the program will exit with an error.

## Building

```
$ git clone https://github.com/nilskohl/mpi-comm-bench
$ mkdir build && cd build

# CPU build
$ cmake ../mpi-comm-bench

# GPU build
$ cmake -DUSE_CUDA=ON ../mpi-comm-bench

$ make
```

## Running

Run with `-h` or `--help` to see available options.

```
Usage: mpi_comm_bench [--msg-size=<msg-size-in-bytes>] [--interval=<interval-in-sec>] [--gpu] [--mem-copy-local]
```

* `--msg-size`: Size of the message to send in bytes.
* `--interval`: Interval in seconds between sending messages.
* `--gpu`: Run on GPU (requires CUDA) - just compiling with CUDA enabled and running without this flag will run on CPU.
* `--mem-copy-local`: Uses `memcpy` or `cudaMemcpy` respectively if invoked with a single process.

### Example output
```
$ mpirun -np 2 ./mpi_comm_bench --msg-size 64000000 --gpu
Ring comm benchmark.
Message size:   64000000 bytes (~64 MB, ~0.064 GB).
Interval:       1 seconds.
GPU mode:       on
Mem copy local: off
Bandwidth (send + recv): min =      1.060 GB/s | max =      1.060 GB/s | avg =      1.060 GB/s || Duration (send + recv): min =    120.738 ms | max =    120.738 ms | avg =    120.738 ms
Bandwidth (send + recv): min =      0.603 GB/s | max =      0.603 GB/s | avg =      0.603 GB/s || Duration (send + recv): min =    212.168 ms | max =    212.169 ms | avg =    212.169 ms
Bandwidth (send + recv): min =     30.673 GB/s | max =     30.679 GB/s | avg =     30.676 GB/s || Duration (send + recv): min =      4.172 ms | max =      4.173 ms | avg =      4.173 ms
Bandwidth (send + recv): min =     28.891 GB/s | max =     28.896 GB/s | avg =     28.894 GB/s || Duration (send + recv): min =      4.430 ms | max =      4.430 ms | avg =      4.430 ms
Bandwidth (send + recv): min =     30.449 GB/s | max =     30.455 GB/s | avg =     30.452 GB/s || Duration (send + recv): min =      4.203 ms | max =      4.204 ms | avg =      4.203 ms```
^C
```


