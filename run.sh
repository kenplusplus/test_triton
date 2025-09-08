#! /bin/bash


CURR_DIR=$(cd $(dirname $0); pwd)

export PYTHONPATH=${CURR_DIR}/src/

cd ${CURR_DIR}/external/ByteMLPerf/byte_micro_perf
python launch.py --task_dir workloads/mocked_model/ --task moe_softmax_topk_triton