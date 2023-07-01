export NCCL_SOCKET_IFNAME=eno1   # 网卡名称要更换为自己的
# 参考：https://www.autodl.com/docs/distributed_training/
# 以下nnodes为多个节点（即实例），nproc_per_node为当前节点上跑几个进程（即几颗GPU），node_rank为第多少个节点，master_addr为master对应实例的网卡ip地址，以上4个变量需要根据实际情况做更改。master_port可以不用调整
# python -m torch.distributed.launch \
#    --nproc_per_node=1 \
#    --nnodes=2 \
#    --node_rank=0 \
#    --master_addr="10.0.0.2" \
#    --master_port=55568 \
#    ddp.py

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_addr="127.0.0.1" \
    ddp.py

