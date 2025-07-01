import os
import trimesh
import numpy as np
import traceback
from maps import MAPS
from multiprocessing import Pool
from multiprocessing.context import TimeoutError as MTE
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


SHREC_CONFIG = {
    'dst_root': './data/SHREC11-MAPS-48-4-split10',
    'src_root': './data/shrec11-split10',
    'n_variation': 10,
    'base_size': 48,
    'depth': 4
}

CUBES_CONFIG = {
    'dst_root': './data/Cubes-MAPS-48-4',
    'src_root': './data/cubes',
    'n_variation': 10,
    'base_size': 48,
    'depth': 4
}

MANIFOLD40_CONFIG = {
    'dst_root': '/data/lcs/test_shapeNet/remesh_after_10000',
    'src_root': '/data/lcs/test_shapeNet/simplify_after_10000',
    'n_variation': 10,
    'base_size': 256,
    'max_base_size': 256,
    'depth': 3
}


def maps_async(obj_path, out_path, base_size, max_base_size, depth, timeout, 
        trial=1, verbose=True):
    if verbose:
        print('[IN]', out_path)

    for _ in range(trial):
        try:
            mesh = trimesh.load(obj_path, process=False)
            maps = MAPS(mesh.vertices, mesh.faces, base_size, timeout=timeout, 
                verbose=verbose)
            if maps.base_size > max_base_size:
                continue
            sub_mesh = maps.mesh_upsampling(depth=depth)
            sub_mesh.export(out_path)
            break
        except Exception as e:
            if verbose:
                traceback.print_exc()
            continue
    else:
        if verbose:
            print('[OUT FAIL]', out_path)
        return False, out_path
    if verbose:
        print('[OUT SUCCESS]', out_path)
    return True, out_path


def make_MAPS_dataset(dst_root, src_root, base_size, depth, n_variation=None, 
        n_worker=1, timeout=None, max_base_size=None, verbose=False):
    '''
    Remeshing a dataset with the MAPS algorithm.
    Parameters
    ----------
    dst_root: str,
        path to a destination directory.
    src_root: str,
        path to the source dataset.
    n_variation:
        number of remeshings for a shape.
    n_workers:
        number of parallel processes.
    timeout:
        if timeout is not None, terminate the MAPS algorithm after timeout seconds.
    References:
        - Lee, Aaron WF, et al. "MAPS: Multiresolution adaptive parameterization of surfaces." 
        Proceedings of the 25th annual conference on Computer graphics and interactive techniques. 1998.
    '''

    if max_base_size is None:
        max_base_size = base_size
    os.makedirs(dst_root,exist_ok=False)

    results = []
    pool = Pool(processes=n_worker)
    for obj_path in Path(src_root).iterdir(): 
        ret = pool.apply_async(
                                    maps_async, 
                                    (str(obj_path), os.path.join(dst_root,obj_path.name), base_size, max_base_size, depth, timeout)
        )
        results.append(ret)  
    if n_worker > 0:
                        try:
                            [r.get(timeout + 1) for r in results]
                            pool.close()
                        except MTE:
                            pass

def make_MAPS_shape(in_path, out_path, base_size, depth):
    in_path = Path(in_path)
    mesh = trimesh.load_mesh(in_path, process=False)
    maps = MAPS(mesh.vertices, mesh.faces, base_size=base_size, timeout=20,verbose=True)
    sub_mesh = maps.mesh_upsampling(depth=depth)
    if sub_mesh.faces.shape[0] == base_size*64:
        sub_mesh.export(out_path)

 
def MAPS_demo1(args):
    '''Apply MAPS to a single 3D model'''
    data_root = Path(args.dataroot)
    output_path = args.output
    pool = Pool(processes=32)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for obj in data_root.iterdir():
        if os.path.exists(os.path.join(output_path,obj.name)):
             continue
        # print(obj)
        try:
            pool.apply_async(
                make_MAPS_shape, 
                (obj, os.path.join(output_path,obj.name), 256, 3)
            )
        except:
            print(obj)
            continue
    pool.close()
    pool.join()


def MAPS_demo2():
    '''Apply MAPS to shapes from a dataset in parallel'''
    config = MANIFOLD40_CONFIG

    make_MAPS_dataset(
        config['dst_root'], 
        config['src_root'], 
        config['base_size'],
        config['depth'],
        n_variation=config['n_variation'], 
        n_worker=64, 
        timeout=100,
        max_base_size=config.get('max_base_size'),
        verbose=True
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    MAPS_demo1(args)
