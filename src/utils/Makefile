matrix:
	@echo "LD_LIBRARY_PATH = $(LD_LIBRARY_PATH)"
	@echo "CUDA_VISIBLE_DEVICES = $(CUDA_VISIBLE_DEVICES)"
	nvcc -O3 -o matrix.out matrix.cu -lineinfo -ccbin /opt/compiler/gcc-10/bin -std=c++17 -I/opt/compiler/cuda-11.1/include -I/opt/compiler/gcc-10/include

sort:
	nvcc -O3 -o matrix.out sort.cu -lineinfo -ccbin /opt/compiler/gcc-10/bin -std=c++17 -I/opt/compiler/cuda-11.1/include -I/opt/compiler/gcc-10/include

scan:
	nvcc -O3 -o matrix.out scan.cu -lineinfo -ccbin /opt/compiler/gcc-10/bin -std=c++17 -I/opt/compiler/cuda-11.1/include -I/opt/compiler/gcc-10/include

reduce:
	nvcc -O3 -o matrix.out reduce.cu -lineinfo -ccbin /opt/compiler/gcc-10/bin -std=c++17 -I/opt/compiler/cuda-11.1/include -I/opt/compiler/gcc-10/include

clean:
	rm -rf ./matrix.out