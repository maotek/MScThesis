#!/usr/bin/env python3

import torch

def main():
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    if not cuda_available:
        return

    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)

    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)

        # Memory usage from torch.cuda.memory_stats()
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)

        print(f"\nGPU {i}: {name}")
        print(f"  Total memory:     {total_mem:.2f} GB")
        print(f"  Allocated memory: {allocated:.2f} GB")
        print(f"  Reserved memory:  {reserved:.2f} GB")

        # Optional separate free memory estimation:
        free_mem = total_mem - reserved
        print(f"  Free memory:      {free_mem:.2f} GB")

if __name__ == "__main__":
    main()
