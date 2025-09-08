

# Test Using Triton to optimize the kernel SoftMaxTopK

## Reference
- https://github.com/bytedance/ByteMLPerf/blob/main/byte_micro_perf/workloads/llm/README.md
- https://github.com/bytedance/ByteMLPerf/blob/main/byte_micro_perf/core/ops/llm_ops.py
- https://github.com/bytedance/ByteMLPerf/blob/main/byte_micro_perf/workloads/mocked_model/moe_softmax_topk.json
- https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/softmax.py

## Install

- OS: Ubuntu 24.04
- Conda: 24.3.0
- Cuda: [12.9](https://developer.nvidia.com/cuda-12-9-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)

    ```
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
    sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.1-575.57.08-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2404-12-9-local_12.9.1-575.57.08-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2404-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-9
    ```
- Torch: ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129```
- Triton: ```pip install triton```

## Run

```
cd external/ByteMLPerf/byte_micro_perf
python launch.py --task_dir workloads/mocked_model/ --task moe_softmax_topk_triton
```

## Modifications

<https://github.com/kenplusplus/ByteMLPerf/commits/ken-triton/>
