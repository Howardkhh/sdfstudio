USER=howardkhh
docker run --gpus all \
            -v /home/${USER}/sdfstudio/data/:/home/user/sdfstudio/data/ \
            -v /home/${USER}/sdfstudio/outputs/:/home/user/sdfstudio/outputs \
            -v /home/${USER}/sdfstudio/exports/:/home/user/sdfstudio/exports/ \
            -v /home/${USER}/sdfstudio/renders/:/home/user/sdfstudio/renders/ \
            -v /home/${USER}/.cache/:/root/.cache/ \
            -p 7007:7007 \
            --rm \
            -it \
            --shm-size=12gb \
            sdfstudio-61