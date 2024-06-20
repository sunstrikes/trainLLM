#pragma once
#include <cuda_runtime.h>

namespace train_llm {

template <typename T>
void debug_display(std::string&& name, T* device_ptr, int len) {
    auto host_ptr = new T[len];
    cudaMemcpy(host_ptr, device_ptr, len * sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << "display gpu_array, name: " << name << ", len: " << len << ", [";
    for (int i = 0; i < len; ++i) {
        std::cout << host_ptr[i] << ", ";
    }
    std::cout << "]" << std::endl;
    delete[] host_ptr;
}

template <typename T>
void debug_display_mat(std::string&& name, T* device_ptr, int M, int N) {
    auto len = M * N;
    auto host_ptr = new T[len];
    cudaMemcpy(host_ptr, device_ptr, len * sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << "display gpu_array, name: " << name << ", len: " << len << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << "[";
        for (int j = 0; j < N; ++j) {
            std::cout << std::noshowpos << std::setw(9) << std::left << host_ptr[i * N + j] << ", ";
        }
        std::cout << "]" << std::endl;
    }
    delete[] host_ptr;
}

template void debug_display_mat<float>(std::string&& name, float* device_ptr, int M, int N);
template void debug_display_mat<int>(std::string&& name, int* device_ptr, int M, int N);
}  // namespace train_llm