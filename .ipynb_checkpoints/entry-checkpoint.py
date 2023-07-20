import os
import json
import socket

if __name__ == "__main__":
   
    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))
    
    #Parse the IP address of the master node in the multiple nodes cluster of SageMaker training.
    master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    master_addr = socket.gethostbyname(master)
    
    
    os.environ['NODE_INDEX'] = str(host_rank)
    os.environ['SM_MASTER'] = str(master)
    os.environ['SM_MASTER_ADDR'] = str(master_addr)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    # backend env config
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['FI_PROVIDER'] = 'efa'
    os.environ['NCCL_PROTO'] = 'simple'
    os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['HCCL_OVER_OFI'] = '1'
    
    
    os.system("wandb disabled")
    
    
    #invoke the torch launcher shell script.
    #Note: we will use the pytorch launcher to launch deepspeed for multi-nodes training.
    #Note: we will use the s5cmd to speed up the uploading model assets to S3.
    os.system("chmod +x ./train.sh")
    os.system("chmod +x ./s5cmd")
    os.system("/bin/bash -c ./train.sh")
    
