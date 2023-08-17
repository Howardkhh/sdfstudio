docker run --gpus all \
            -v /home/howardkhh/sdfstudio/data/:/home/user/sdfstudio/data/ \
            -v /home/howardkhh/sdfstudio/outputs/:/home/user/sdfstudio/outputs \
            -v /home/howardkhh/sdfstudio/exports/:/home/user/sdfstudio/exports/ \
            -v /home/howardkhh/.cache/:/root/.cache/ \
            -p 7007:7007 \
            --rm \
            -it \
            --shm-size=12gb \
            sdfstudio-61