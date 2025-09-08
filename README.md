

# Test Using Triton to optimize the kernel SoftMaxTopK

## Reference
- https://github.com/bytedance/ByteMLPerf/blob/main/byte_micro_perf/workloads/llm/README.md
- https://github.com/bytedance/ByteMLPerf/blob/main/byte_micro_perf/core/ops/llm_ops.py
- https://github.com/bytedance/ByteMLPerf/blob/main/byte_micro_perf/workloads/mocked_model/moe_softmax_topk.json
- https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/softmax.py

## Install

- OS: Ubuntu 24.04
- Cuda: [12.9](https://developer.nvidia.com/cuda-12-9-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)