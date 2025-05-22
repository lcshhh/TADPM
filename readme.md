# TADPM：Automatic Tooth Arrangement with Joint Features of Point and Mesh Representations via Diffusion Probabilistic Models

This is the PyTorch implementation of our TADPM.

### Requirements

To install python requirements:

```shell
pip install -r requirements.txt
```

To install chamfer distance:

```shell
cd chamfer_dist
python setup.py install
```

To install manifold, please refer to https://github.com/ZhaoHengJiang/MeshReconstruction/tree/main/Manifold

### Data pre-process

- To get single tooth mesh files:

```shell
bash scripts/get_mesh.sh
```

- To get pointcloud files:

```shell
bash scripts/get_pointcloud.sh
```

- To get remesh files:

```shell
bash scripts/remesh.sh
```

- To get ground truth rotation parameters, you can either use the **Kabsch algorithm** to obtain an accurate rotation matrix(If the points on your pre- and post-orthodontic jaw models can be matched one by one) or use **ICP algorithm** to obtain an approximate rotation matrix.

  You need to save the rotation parameters for each dental model in a separate pkl file. Each file should contain a tensor of shape [32, 3, 3], where each [3, 3] matrix corresponds to the rotation matrix of an individual tooth. **All the files should be placed under the paramroot.**

  Here is an example of the Kabsch algorithm:

  ```python
  # you need to implement the remaining code
  def kabsch_algorithm(A, B):
      assert A.shape == B.shape
      
      A_mean = A - np.mean(A, axis=0)
      B_mean = B - np.mean(B, axis=0)
  
      H = np.dot(A_mean.T, B_mean)
  
      U, S, Vt = np.linalg.svd(H)
  
      R = np.dot(Vt.T, U.T)
  
      if np.linalg.det(R) < 0:
          Vt[2, :] *= -1
          R = np.dot(Vt.T, U.T)
  
      return R
  ```

  For the ICP algorithm, you can directly use the interface provided by the `trimesh` library.

### Pretrain

- To pretrain MeshMAE:

```shell
bash scripts/pretrain.sh
```

You can also refer to https://github.com/liang3588/MeshMAE

### Train

- To train the TADPM model:

```shell
bash scripts/train.sh
```

When training TADPM, you should set the path to pretrained MeshMAE model checkpoint.

### Test

- To visualize TADPM's results, run:

```
bash scripts/get_result.sh
```