export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export MASTER_PORT=25643 # Avoid multiple programs competing for the same port.

corr_scaler=0.005 # The correction coefficient.
path="samples" # The save path for the generated images.

torchrun --standalone --nnodes=1 --nproc_per_node=5 generate.py \
    --outdir=${path} --batch 2000 --steps=13 --seeds=0-49999 --solver=euler  \
    --corr_scaler ${corr_scaler} \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

torchrun --standalone --nproc_per_node=1 fid.py calc --images=${path} \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz --scaler=${corr_scaler}


