export LC_ALL="C"
export LD_LIBRARY_PATH=./output/so:/opt/compiler/cuda-11.1/lib64:/opt/compiler/cudnn/cuda11.1/cudnn-8.1.1/lib64:/usr/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=1
/opt/compiler/cuda-11.1/bin/nv-nsight-cu-cli -f --target-processes all -o profile --set full --devices 0 ./matrix.out
