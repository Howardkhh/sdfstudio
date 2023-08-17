docker build \
    --build-arg CUDA_ARCHITECTURES=61 \
    --tag sdfstudio-61 \
    --file Dockerfile .