#!/bin/bash
UID=$(id -u)
GID=$(id -g)
UNAME=$(whoami)

. constants.env

docker run -it -d --rm --runtime=nvidia \
        --name psa_bev_lane_det_$UNAME \
        --cpus $CPUS \
        -v $PWD:/workspace \
        -v $DATADIR:/data \
        -v $DATADIR2:/data2 \
        -v $DATADIR3:/data3 \
        -v $DATADIRL:/dataL \
        -v $HOME:/vscode_home \
        -e "HOME=/vscode_home" \
        -p $PORTS:1237-1239 \
        -p $SSH_PORT:22 \
        -w /workspace \
        --shm-size="256g"\
        psa_bev_lane_det:latest

