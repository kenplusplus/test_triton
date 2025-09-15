#! /bin/bash


CURR_DIR=$(cd $(dirname $0); pwd)
KERNEL="torch"

export PYTHONPATH=${CURR_DIR}/src/
cd ${CURR_DIR}/external/ByteMLPerf/byte_micro_perf

while getopts ":k:" opt; do
    case $opt in
        k)
            KERNEL="$OPTARG";;
        \?)
            echo "Invalid option: -"$OPTARG"" >&2
            exit 1;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

case $KERNEL in
    torch)
        python launch.py --task_dir workloads/mocked_model/ --task moe_softmax_topk
        ;;
    triton)
        rm ${CURR_DIR}/src/triton_kernel.py
        ln -s ${CURR_DIR}/src/triton_kernel_orig.py ${CURR_DIR}/src/triton_kernel.py
        python launch.py --task_dir workloads/mocked_model/ --task moe_softmax_topk_triton
        ;;
    triton_fp16)
        rm ${CURR_DIR}/src/triton_kernel.py
        ln -s ${CURR_DIR}/src/triton_kernel_fp16.py ${CURR_DIR}/src/triton_kernel.py
        python launch.py --task_dir workloads/mocked_model/ --task moe_softmax_topk_triton
        ;;
    triton_tile)
        rm ${CURR_DIR}/src/triton_kernel.py
        ln -s ${CURR_DIR}/src/triton_kernel_tile.py ${CURR_DIR}/src/triton_kernel.py
        python launch.py --task_dir workloads/mocked_model/ --task moe_softmax_topk_triton
        ;;
esac
