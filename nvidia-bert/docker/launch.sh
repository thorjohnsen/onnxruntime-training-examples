#!/bin/bash
  
CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"all"}
DOCKER_BRIDGE=${3:-"host"}

nvidia-docker run -it --rm \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /mnt/dldata/cforster/mlperf/dataset/phase1_checkpoint:/workspace/phase1 \
  -v /mnt/dldata/cforster/mlperf/dataset/hdf5/2048_shards:/data/512 \
  -v $PWD/results:/results \
  -v $PWD:/workspace/current_dir \
  --workdir=/workspace/bert \
  gitlab-master.nvidia.com:5005/tjohnsen/mlperf_logging:onnxruntime-bert.20.06.ampere $CMD
#  -v $PWD:/workspace/bert \
