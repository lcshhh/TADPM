import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
from multiprocessing import Pool
import trimesh
import vedo
import argparse
import copy
import pyfqmr
import csv

def manifold(obj_path,dataroot,output_root_manifold):
    if os.path.exists(output_root_manifold + '/' + obj_path.name):
        return
    mesh = vedo.Mesh(str(obj_path))
    oface_number = len(mesh.cells)
    mface_number = oface_number * 1.2
    # mface_number = 20000
    commandm = '../Manifold/build/manifold ' + str(
        obj_path) + ' ' + output_root_manifold + '/' + obj_path.name + ' ' + str(int(mface_number))
    try:
        status1 = os.system(commandm)
    except:
        if status1 != 0:
            raise Exception('wrong, command=%s, status=%s' % (commandm, status1))

def simplify(obj_path,output_root_simplify,output_root_manifold):
    commands = '../Manifold/build/simplify -i ' + output_root_manifold + '/' + obj_path.name + ' -o ' + output_root_simplify + '/' + obj_path.name + ' -m -f ' + str(
                256)
    try:
        status = os.system(commands)
    except:
        if status != 0:
            raise Exception('wrong, command=%s, status=%s' % (commands, status))

def quad_simplify(obj_path,output_root_simplify,output_root_manifold):
    if os.path.exists(os.path.join(output_root_simplify,obj_path.name)):
        return
    mesh = trimesh.load_mesh(os.path.join(output_root_manifold,obj_path.name))
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
    mesh_simplifier.simplify_mesh(target_count = 256, aggressiveness=7, preserve_border=True, verbose=10)
    vertices, faces, normals = mesh_simplifier.getMesh()
    mesh.vertices = vertices
    mesh.faces = faces
    mesh.export(os.path.join(output_root_simplify,obj_path.name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--manifold_before', type=str, required=True)
    parser.add_argument('--simplify_before', type=str, required=True)
    args = parser.parse_args()

    root = args.dataroot
    dataroot = Path(root)
    output_root_manifold = args.manifold_before
    output_root_simplify = args.simplify_before
    if not os.path.exists(output_root_manifold):
        os.mkdir(output_root_manifold)
    if not os.path.exists(output_root_simplify):
        os.mkdir(output_root_simplify)
    pool = Pool(processes=16)
    for obj_path in dataroot.iterdir():
        if obj_path.is_file():    
            pool.apply_async(
                quad_simplify,
                (obj_path,output_root_simplify,output_root_manifold)
            )
    pool.close()
    pool.join()
