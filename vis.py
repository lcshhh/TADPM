import numpy as np
import open3d as o3d
def read_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points)
    return xyz 

def write_pointcloud(points, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)

def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

xml_head = \
"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.013"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-1"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]
xml_segments = [xml_head]

# pcl = np.load('chair_pcl.npy')
import torch
# torch.manual_seed(42)
# pcl1 = read_pointcloud('/data3/leics/dataset/synthetic/half_merged_pcd_after/10_upper.ply')
# pcl1 = read_pointcloud('/data3/leics/dataset/synthetic/merged_pcd_after/1.ply')
pcl = read_pointcloud('/data3/leics/dataset/test_single/zuhe/full.ply')
# pcl = read_pointcloud('/data3/leics/dataset/synthetic/merged_pcd_before/1.ply')
# pcl2 = 0.5 * torch.randn(pcl1.shape[0],3).numpy()
# para = 0.9
# pcl = para * pcl1 + (1-para) * pcl2
# 定义旋转角度（以弧度为单位）
theta = np.radians(60)  # upper为-30，正面为60, lower为150

# 定义绕 x 轴旋转 30 度的旋转矩阵
rotation_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])

# rotation_y = np.array([
#     [-1, 0, 0],
#     [0, 1, 0],
#     [0, 0, -1]
# ])
theta = np.radians(45)  # 30 度转换为弧度

# 定义绕 y 轴旋转 30 度的旋转矩阵
rotation_y = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])

# theta = np.radians(30)  # 30 度转换为弧度

# # 定义绕 z 轴旋转 30 度的旋转矩阵
# rotation_z_30 = np.array([
#     [np.cos(theta), -np.sin(theta), 0],
#     [np.sin(theta), np.cos(theta), 0],
#     [0, 0, 1]
# ])
pcl = pcl @ rotation_x
pcl = pcl @ rotation_y
# pcl = pcl @ rotation_z_30.T
# pcl = pcl @ rotation_y.T
pcl = standardize_bbox(pcl, pcl.shape[0])
pcl = pcl[:,[2,0,1]]
pcl[:,0] *= -1
pcl[:,2] += 0.0125

for i in range(pcl.shape[0]):
    # color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
    color = np.array([0.5, 0.7, 1.0])
    xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
xml_segments.append(xml_tail)

xml_content = str.join('', xml_segments)

with open('test.xml', 'w') as f:
    f.write(xml_content)
import os
os.system('mitsuba test.xml')


