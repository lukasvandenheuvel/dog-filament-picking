import matplotlib
import yaml
import random
import os
import numpy as np
from custom_dog_picking_par import read_data,normalize,skeletonize,detect_lines,prune_lines
from skimage import filters,morphology,transform,draw,io
import napari

path_to_yaml = './params.yaml'

with open(path_to_yaml, 'r') as file:
    params = yaml.safe_load(file)

root = params['root']
path_to_micrographs_star = params['path_to_micrographs_star']
rescale             = params['rescale']
sigma_view          = params['sigma_view']
dog_low_sigma       = params['dog_low_sigma']
ridge_sigmas        = params['ridge_sigmas']
ridge_smoothing     = params['ridge_smoothing']
ridge_threshold     = params['ridge_threshold']
min_length          = params['min_length']
hough_line_length   = params['hough_line_length']
hough_line_gap      = params['hough_line_gap']
max_angle           = params['max_angle']
max_distance        = params['max_distance']

with open(os.path.join(root,path_to_micrographs_star),'r') as f:
    lines = f.readlines()
    mrcfiles = [f.split()[0] for f in lines if '.mrc' in f]

rel_mrc_path = random.choice(mrcfiles)
mrc_path = os.path.join(root, rel_mrc_path)
data,ps = read_data(mrc_path,rescale)
dog = filters.difference_of_gaussians( normalize(data), low_sigma=dog_low_sigma/ps )
skel = skeletonize(dog,ps,min_length,ridge_threshold)
line_coords = detect_lines(skel,ps,hough_line_length,hough_line_gap)
pruned_line_coords = prune_lines(line_coords, ps, min_length, max_angle, max_distance)

lines = [[[coord[0][1],coord[0][0]],[coord[1][1],coord[1][0]]] for coord in line_coords]
pruned_lines = [[[coord[0][1],coord[0][0]],[coord[1][1],coord[1][0]]] for coord in pruned_line_coords]

cmap = matplotlib.cm.get_cmap('Spectral')
viewer = napari.Viewer()
image_layer = viewer.add_image(filters.gaussian( data, sigma=sigma_view/ps ))
image_layer = viewer.add_image(dog)
# image_layer = viewer.add_image(ridges)
# image_layer = viewer.add_image(ridges>ridge_threshold)
# image_layer = viewer.add_image(skel)
image_layer = viewer.add_shapes(lines,shape_type='line',edge_width=10)
image_layer = viewer.add_shapes(pruned_lines,shape_type='line',edge_width=10,edge_color=cmap(np.linspace(0,1,len(pruned_lines))))
napari.run()