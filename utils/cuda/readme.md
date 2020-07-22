# Determine your CUDA compute capability

This script helps you to determine your CUDA compute capability.
I.e. which fancy algorithms your GPU supports.

## How to?

Just run the script `get_cuda_sm.sh` and it outputs a number which is the CUDA compute capability.

It automatically compiles a small c program which uses the CUDA library to query the device capabilities
(it seems that this unfortunately can't be done through a command line tool such as nvidia-smi ...).

## Where to put it?

In `src/makefile` where you specify the definitions for your FARGO_ARCH, extend the `CUDAOPT_LINUX` variable by `-arch=sm_XX` where `XX` is the result of this script.