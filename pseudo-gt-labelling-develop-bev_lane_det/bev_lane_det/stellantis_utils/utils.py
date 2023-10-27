
import os
import csv
import logging
from re import X
import struct
from pathlib import Path
import json

import pdb

LOG = logging.getLogger(__name__)

_THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = _THIS_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"

def loadOdomList(path,sep=' '):
    with open(path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=sep)
        line_count = 0
        timestamps=[]
        values=[]
       
        for row in csv_reader:
            timestamps.append(int(row[0]))
            values.append(float(row[1]))
            line_count += 1
        
        LOG.debug(f'Processed {line_count} lines.')

        return timestamps,values

def extract_can_data(can_file):
    timestamp_list=[]
    accel_list=[]
    speed_list=[]
    yawrate_list=[]
    assert os.path.isfile(can_file), "Cannot find CAN file {}".can_file
    with open(can_file, "rb") as f:
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

    # Write to csv files
    out_dir = os.path.dirname(os.path.abspath(can_file))
    outfileaccel = os.path.join(out_dir, 'accel.txt')
    outfilespeed = os.path.join(out_dir, 'speed.txt')
    outfileAngularRate = os.path.join(out_dir, 'angularRate.txt')

    writeAccelaration(outfileaccel, timestamp_list, accel_list)
    writeSpeed(outfilespeed, timestamp_list, speed_list)
    writeAngularRate(outfileAngularRate, timestamp_list, yawrate_list)

    return

def writeAccelaration(path, ts_accel, accel):
        with open(str(path), mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for acc_elt,ts in zip(accel, ts_accel):
                writer.writerow([str(ts),str(acc_elt)])

def writeSpeed(path, ts_speed, speed):
    with open(str(path), mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for speed_elt,ts in zip(speed,ts_speed):
            writer.writerow([str(ts), str(speed_elt)])


def writeAngularRate(path, ts_rot, rot):
    with open(str(path), mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for rot_elt,ts in zip(rot, ts_rot):
            writer.writerow([str(ts), str(rot_elt)])

def loadImageList(path):
    with open(path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=' ')
        line_count = 0
        
        image_list=[]
        timestamps=[]
        for row in csv_reader:
            timestamps.append(int(row[0]))
            image_list.append(str(replace_var(row[1])))
            line_count += 1
        
        LOG.debug(f'Processed {line_count} lines.')

        return image_list,timestamps

def replace_var(path): # TODO: usefull ?
    """Replace {VAR} in path string and return a Path."""
    if path is not None:
        path = Path(str(path).replace("{DATA_DIR}", str(DATA_DIR)))
    else:
        path = None
    return path

def load_persformer3d_lanes(json_file):

    if not os.path.exists(json_file):
        LOG.error("File {} not found.".format(json_file))
        return None
    with open(json_file) as file:
        data = json.load(file)
    
    # Convert coordinates so as to match the PSA convention 
    for dict in data['pred']:
        x = dict["y_3d"].copy()
        y = dict["x_3d"].copy()
        dict["x_3d"] = x
        dict["y_3d"] = y

def load_bev_lane_det_lanes(json_file):

    if not os.path.exists(json_file):
        LOG.error("File {} not found.".format(json_file))
        return None
    with open(json_file) as file:
        data = json.load(file)
    
    # Convert coordinates so as to match the PSA convention 
    out = {
        "pred":data[0],
        "gt": data[1]
        }

    return out
