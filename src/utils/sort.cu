#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

#include "time_recorder.h"
#include "util.h"

namespace train_llm {

/********************* bitonic sort *********************/

__global__ void bitonic_sort_step(int* dev_values, int j, int k) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i) {
        if ((i & k) == 0) {
            /* Sort ascending */
            if (dev_values[i] > dev_values[ixj]) {
                /* exchange(i,ixj); */
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i & k) != 0) {
            /* Sort descending */
            if (dev_values[i] < dev_values[ixj]) {
                /* exchange(i,ixj); */
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

/********************* radix sort *********************/

}  // namespace train_llm

using namespace train_llm;
void test_bitonic_sort() {
    int* dev_values;
    const int NUM_VALS = 16;
    size_t size = NUM_VALS * sizeof(int);
    std::vector<int> values(NUM_VALS);
    random_int_tensor(values.data(), 16, 16);
    cudaMalloc((void**)&dev_values, size);
    cudaMemcpy(dev_values, values.data(), size, cudaMemcpyHostToDevice);

    int grid_size = 1;
    int THREADS = 256;
    int j, k;
    /* Major step */
    debug_display_mat("step_res:", dev_values, 1, 16);
    for (k = 2; k <= NUM_VALS; k <<= 1) {
        /* Minor step */
        for (j = k >> 1; j > 0; j = j >> 1) {
            bitonic_sort_step<<<grid_size, THREADS>>>(dev_values, j, k);
            std::cout << "step: " << j << " " << k << std::endl;
            debug_display_mat("step_res:", dev_values, 1, 16);
        }
    }
    cudaMemcpy(values.data(), dev_values, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
}
int main() {
    // test_transpose();
    test_bitonic_sort();
}