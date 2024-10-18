import matplotlib
import yaml
import random
import os
import sys
import numpy as np
from custom_dog_picking_par import read_data,normalize,skeletonize,detect_lines,prune_lines,keep_lines_with_helical_rise,radial_psd_distribution,butter_lowpass_filter,psd_2D
from skimage import filters,morphology,transform,draw,io
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import napari
import pandas as pd
from scipy import signal


path_to_yaml = './params.yaml'

with open(path_to_yaml, 'r') as file:
    params = yaml.safe_load(file)

root = params['root']
path_to_micrographs_star = params['path_to_star']
rescale             = params['rescale']
sigma_view          = params['sigma_view']
sigma_background    = params['sigma_background']
dog_sigmas          = params['dog_sigmas']
ridge_threshold     = params['ridge_threshold']
min_length_skel     = params['min_length_skel']
hough_line_length   = params['hough_line_length']
hough_line_gap      = params['hough_line_gap']
max_angle           = params['max_angle']
max_distance        = params['max_distance']
min_length          = params['min_length']
ellipse_kernel_size = params['ellipse_kernel_size']
ellipse_radius      = params['ellipse_radius']
ellipse_theta       = params['ellipse_theta']
edge_percentage     = params['edge_percentage']
check_helix_fft     = params['check_helix_fft']
num_freq_bins       = params['num_freq_bins']
butterfilter_order  = params['butterfilter_order']
butterfilter_cutoff = params['butterfilter_cutoff']
peak_prominence     = params['peak_prominence']
helical_rise_range  = params['helical_rise_range']

def remove_lines_on_edge(line_coords, data, edge_percentage=0.1):
    # Remove lines too close to the image edge
    pruned_coords = []
    height,width = data.shape
    for coord in line_coords:
        if (coord[0][0]<(edge_percentage*width)) and (coord[1][0]<(edge_percentage*width)): # vertical too close to left
            continue
        if (coord[0][0]>((1-edge_percentage)*width)) and (coord[1][0]>((1-edge_percentage)*width)): # vertical too close to right
            continue
        if (coord[0][1]<(edge_percentage*height)) and (coord[1][1]<(edge_percentage*height)): # horizontal too close to top
            continue
        if (coord[0][1]>((1-edge_percentage)*height)) and (coord[1][1]>((1-edge_percentage)*height)): # horizontal too close to bottom
            continue
        pruned_coords.append(coord)
    return pruned_coords

with open(os.path.join(root,path_to_micrographs_star),'r') as f:
    lines = f.readlines()
    mrcfiles = [os.path.split(f.split()[0])[1] for f in lines if '.mrc' in f]

mrcfile_choice = random.choice(mrcfiles)
mrc_path = os.path.join(root, mrcfile_choice)
data,data_rescale,ps = read_data(mrc_path,rescale)
height,width = data.shape
background_subtract = data_rescale - filters.gaussian( data_rescale, sigma=sigma_background/ps )
dog = filters.difference_of_gaussians( normalize(background_subtract), low_sigma=dog_sigmas[0]/ps )
for sigma in dog_sigmas[1:]:
    dog = filters.difference_of_gaussians( normalize(dog), low_sigma=sigma/ps )

# Convolve with inverted ellipse
ellipse = np.zeros((int(ellipse_kernel_size/ps),int(ellipse_kernel_size/ps)))
rr,cc = draw.ellipse(int(ellipse_kernel_size/ps/2),int(ellipse_kernel_size/ps/2),int(ellipse_radius/ps/2),int(ellipse_kernel_size/ps/2))
ellipse[rr,cc] = 1

line_coords = []
for angle in np.arange(0,180,ellipse_theta):
    ellipse_rot = transform.rotate(ellipse,angle=angle,resize=False)
    ellipse_rot = ellipse_rot / np.sum(ellipse_rot)
    cnv = convolve(dog,ellipse_rot)
    skel = skeletonize(cnv,ps,min_length_skel,ridge_threshold)

    line_coords += detect_lines(skel,ps,hough_line_length,hough_line_gap)

pruned_line_coords = prune_lines(line_coords, ps, min_length, max_angle, max_distance)
if check_helix_fft:
    pruned_line_coords = keep_lines_with_helical_rise(data,pruned_line_coords,ps,rescale,ellipse_radius,helical_rise_range,num_freq_bins,butterfilter_cutoff,butterfilter_order,peak_prominence)
pruned_line_coords = remove_lines_on_edge(pruned_line_coords, data_rescale, edge_percentage=edge_percentage)

# For visualization
final_lines = [[[coord[0][1],coord[0][0]],[coord[1][1],coord[1][0]]] for coord in pruned_line_coords]
lines = [[[coord[0][1],coord[0][0]],[coord[1][1],coord[1][0]]] for coord in line_coords]

cmap = matplotlib.cm.get_cmap('Spectral')
viewer = napari.Viewer()

image_layer = viewer.add_image(filters.gaussian( data_rescale, sigma=sigma_view/ps ))
image_layer = viewer.add_image(filters.gaussian( background_subtract, sigma=sigma_view/ps ))
image_layer = viewer.add_image(dog)
image_layer = viewer.add_image(ellipse)
image_layer = viewer.add_image(cnv)
image_layer = viewer.add_image(skel)

# image_layer = viewer.add_image(ridges)
# image_layer = viewer.add_image(ridges>ridge_threshold)
# image_layer = viewer.add_image(skel)
image_layer = viewer.add_shapes(lines,shape_type='line',edge_width=3)
image_layer = viewer.add_shapes(final_lines,shape_type='line',edge_width=3,edge_color=cmap(np.linspace(0,1,len(final_lines))))
napari.run()