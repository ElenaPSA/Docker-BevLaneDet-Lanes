#!/bin/bash
UID=$(id -u)
GID=$(id -g)
UNAME=$(whoami)
docker exec -it --user $UID:$GID  psa_bev_lane_det_$UNAME bash
