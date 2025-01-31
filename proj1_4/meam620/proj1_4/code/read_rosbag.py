import numpy as np
import pandas as pd
from matplotlib.lines import Line2D 

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import bagpy
from bagpy import bagreader   # this contains a class that does all the hard work of reading bag files

def output():
    bag_name = '/home/yihsuan/meam620_ws/src/proj1_4/meam620/proj1_4/util/map1_final.bag'
    b = bagreader(bag_name)

    #print(b.topic_table)
    #print(b.topics)

    csvfiles = {}     # To avoid mixing up topics, we save each topic as an individual csv file, since some topics might have the same headers!
    for t in b.topics:
        data = b.message_by_topic(t)
        csvfiles[t] = data

    # The topic "odom" contains all the state information we need
    state = pd.read_csv(csvfiles['odom'])

    # Here we are extracting time and subtracting the start time of the .bag file
    time = state['Time'] - b.start_time

    # Position
    x = state['pose.pose.position.x']
    y = state['pose.pose.position.y']
    z = state['pose.pose.position.z']

    # Velocity
    xdot = state['twist.twist.linear.x']
    ydot = state['twist.twist.linear.y']
    zdot = state['twist.twist.linear.z']

    # Angular Velocity (w.r.t. body frames x, y, and z)
    wx = state['twist.twist.angular.x']
    wy = state['twist.twist.angular.y']
    wz = state['twist.twist.angular.z']

    # Orientation (measured as a unit quaternion)
    qx = state['pose.pose.orientation.x']
    qy = state['pose.pose.orientation.y']
    qz = state['pose.pose.orientation.z']
    qw = state['pose.pose.orientation.w']

    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    z = np.array(z).reshape(-1,1)
    new = np.hstack((x, y, z))
    #print(new.shape)

    # Compute the differences between consecutive points
    #differences = np.diff(new, axis=0)

    # Compute the Euclidean distance for each set of differences
    #distances = np.sqrt(np.sum(differences**2, axis=1))
    
    distances = np.sum(np.linalg.norm(np.diff(new, axis=0),axis=1))

    # Sum up the distances to get the total distance
    total_distance = np.sum(distances)

    print("Actual Total distance:", total_distance)
    return new



