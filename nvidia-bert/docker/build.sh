#!/bin/bash
docker build --network=host . --rm --pull --no-cache -t gitlab-master.nvidia.com:5005/tjohnsen/mlperf_logging:onnxruntime-bert.20.06.ampere
