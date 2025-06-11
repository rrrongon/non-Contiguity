import torch
import time

def benchmark_contiguity_switch_cpu(size):
    # Generate a non-contiguous tensor of specified size on CPU
    non_contiguous_tensor_cpu = torch.randn(size)

    # Start timer
    start_time = time.time()

    # Convert the non-contiguous tensor to a contiguous one
    contiguous_tensor_cpu = non_contiguous_tensor_cpu.contiguous()

    # End timer
    end_time = time.time()

    # Calculate the time taken for switching to contiguity
    time_taken = end_time - start_time

    return time_taken

def main_cpu():
    tensor_sizes = [1024, 1024*100, 1024*1024, 1024*1024*10, 1024*1024*100, 1024*1024*500]
    for size in tensor_sizes:
        time_taken = benchmark_contiguity_switch_cpu(size)
        print(f"Tensor size: {size} bytes, Time taken: {time_taken:.6f} seconds")

if __name__ == "__main__":
    main_cpu()

