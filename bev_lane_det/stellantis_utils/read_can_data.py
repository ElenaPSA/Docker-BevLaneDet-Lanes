import numpy as np
import struct
import pdb

def extract_ego_data(file):

    timestamp_list=[]
    accel_list=[]
    speed_list=[]
    yawrate_list=[]

    max_iter = 2000
    iter = 0
    with open(file, "rb") as f:
        while True:
            bytes = f.read(8)
            if not bytes:
                break
            timestamp=struct.unpack('<q', bytes)[0]
            bytes = f.read(8)
            accel=struct.unpack('<d', bytes)[0]
            bytes = f.read(8)
            speed=struct.unpack('<d', bytes)[0]
            bytes = f.read(8)
            yaw=struct.unpack('<d', bytes)[0]
            
            timestamp_list.append(timestamp)
            accel_list.append(accel)
            speed_list.append(speed)
            yawrate_list.append(yaw)

            iter += 1
            if iter > max_iter:
                break

    return timestamp_list,accel_list,speed_list,yawrate_list

if __name__ == '__main__':
    file='/data2/PSA_2022/DataEcon/20220106_103504_Rec_JLAB09/CanEGO.raw'
    timestamp_list,accel_list,speed_list,yawrate_list=extract_ego_data(file)
    
    print(yawrate_list[0:1000])
