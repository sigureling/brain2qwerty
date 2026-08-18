#!/bin/bash
# 通过 rjob 提交 brain2qwerty v2 训练任务（运行 script/brain2qwerty_v2/train.sh）
# 循环提交 seed_list 里每个种子的实验

seed_list='42 3407 1024'
namespace=brainllm

for seed in $seed_list
do
    job_name="brain2qwerty-v2-train-seed${seed}-$(date +%Y%m%d-%H%M%S)"
    rjob submit \
        --name=$job_name \
        --gpu=4 \
        --memory=480000 \
        --cpu=64 \
        --charged-group=${namespace}_gpu \
        --namespace=ailab-${namespace} \
        --private-machine=group \
        --store-host-nvme \
        --custom-resources rdma/mlnx_shared=8 \
        --custom-resources brainpp.cn/fuse=1 \
        --custom-resources mellanox.com/mlnx_rdma=1 \
        -e GROUP=${namespace}_gpu \
        -e DISTRIBUTED_JOB=true \
        -e SEED=$seed \
        --termination-grace-period-seconds 600 \
        --image=registry.h.pjlab.org.cn/ailab-brainllm/xiaoqinfan-workspace:brainomni-deepspeed-20260727 \
        --mount=gpfs://gpfs1/brainllm-share:/mnt/shared-storage-user/brainllm-share \
        --mount=gpfs://gpfs1/xiaoqinfan:/mnt/shared-storage-user/xiaoqinfan \
        -- bash -exc /mnt/shared-storage-user/xiaoqinfan/brain2qwerty/script/brain2qwerty_v2/train.sh
done
