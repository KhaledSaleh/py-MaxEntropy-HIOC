from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Dependency imports
import os
import numpy as np
import re
from tqdm import tqdm
from xml.dom import minidom


def numerical_sort(value):
    """Sort the parsed files from disk numerically."""
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def walk_dir(folder, ext):
    """Walk through all files.ext in a directory."""
    for dirpath, dirs, files in os.walk(folder):
        for filename in sorted(files, key=numerical_sort):
            if filename.endswith(ext):
                yield os.path.abspath(os.path.join(dirpath, filename))


def total_files_counter(inp_path, ext):
    """Count the number of files.ext in a given input path directory."""
    filecounter = 0
    for dirpath, dirs, files in os.walk(inp_path):
        for filename in files:
            if filename.endswith(ext):
                filecounter += 1
    return filecounter


def load_demo_trajs(dir_path):
    """Load the agent demonstrated trajectories from disk.

    Args:
        dir_path: the path to the directory contains the trajectory data
    Returns:
        trajs: List of the agent demonstrated trajectories, each item in the
        list is a tuple of each (x,y) pos of each trajectory.
    """
    # returned list of demonstrated trajectories
    trajs = []
    # Check whether the input txt files path exists or not.
    if not os.path.exists(dir_path):
        print (dir_path + ' is not a valid path')
        exit(-1)
    # Count the total number of *.txt files in the dir_path
    file_counter = total_files_counter(dir_path, '.txt')
    # Check whether there are files avaialable in the dir_path
    if not file_counter:
        print (dir_path + ' contains no files')
        exit(-1)
    for txtFile in tqdm(walk_dir(dir_path, '.txt'), total=file_counter,
                        desc='Parsing trajectory files'):
        trajectory = np.loadtxt(txtFile, dtype=int)
        traj_steps = []
        for i in range(trajectory.shape[0]):
            traj_steps.append((trajectory[i, 1], trajectory[i, 2]))
        trajs.append(traj_steps)
    # returned trajectories
    return trajs


def load_feat_maps(dir_path):
    """Load the feature maps realted to each trajectory from disk.

    Args:
        dir_path: the path to the directory contains the feature maps data
    Returns:
        List of feature maps, each item in the list is
        a 2D numpy array of (state space size, number of features).
    """
    # returned list of demonstrated trajectories
    feat_maps = []
    # Check whether the input txt files path exists or not.
    if not os.path.exists(dir_path):
        print (dir_path + ' is not a valid path')
        exit(-1)
    # Count the total number of *.txt files in the dir_path
    file_counter = total_files_counter(dir_path, '.xml')
    # Check whether there are files avaialable in the dir_path
    if not file_counter:
        print (dir_path + ' contains no files')
        exit(-1)
    for txtFile in tqdm(walk_dir(dir_path, '.xml'), total=file_counter,
                        desc='Parsing feature maps'):
        doc = minidom.parse(txtFile)
        nodes = doc.getElementsByTagName('data')
        feat_map_accum = []
        for idx, node in enumerate(nodes):
            # exclude the output of the tracker features 
            if idx > 36:
                break
            # remove new line character
            data = node.firstChild.nodeValue.replace('\n', '')
            # convert it to a list of string
            data = data.split(' ')
            # filter out empty elements
            data = filter(None, data)
            feat_map_accum.append(data)
        # 2D numpy array (feature map size, number of features)
        feat_map_accum_npy = np.array(feat_map_accum, dtype=float)
        feat_map_accum_npy = feat_map_accum_npy.transpose()
        # append to the total list of feature maps
        feat_maps.append(feat_map_accum_npy)
    # return the total feature maps
    return feat_maps
    
