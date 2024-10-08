import os
import mrcfile
import math
import napari
import numpy as np
import pandas as pd
import networkx as nx
from skimage import filters,morphology,transform,draw,io
from pathlib import Path
import argparse

indx = 9
rescale = 0.5
sigma_view = 10

# Ridge detection
dog_low_sigma = 50      # low sigma for DoG in angstroms. Should be roughly the radius of the fibril
ridge_sigmas = [60,70]  # 
ridge_smoothing = 10
ridge_threshold = -0.006
min_length = 150        # Minimal fibril length in angstrom

# Hough transform
hough_line_length = 40
hough_line_gap = 17

# Line pruning
max_angle = 20
max_distance = 50

def read_data(path,rescale):
    with mrcfile.open(path) as mrc:
        data = transform.rescale(mrc.data, rescale)
        ps = mrc.voxel_size['x'].item() / rescale
    return data,ps

def normalize(img):
    return (img - img.mean()) / img.std()

def detect_ridges(data,ps,ridge_sigmas,ridge_smoothing,):
    ridges = filters.meijering(data,[s/ps for s in ridge_sigmas])
    return filters.gaussian(ridges, sigma=ridge_smoothing/ps)

def skeletonize(ridges,ps,min_length,ridge_threshold):
    skel = morphology.skeletonize(ridges<ridge_threshold)
    return morphology.remove_small_objects(skel, min_size=min_length/ps, connectivity=2)

def detect_lines(skel,ps,line_length,line_gap):
    return transform.probabilistic_hough_line(skel,line_length=int(line_length/ps), line_gap=int(line_gap/ps))

def find_close_line_pairs(line_coords,max_distance):
    # Make vectors of x and y coordinates of starting and ending positions of each line
    x_strt = np.array([crd[0][0] for crd in line_coords]).reshape(-1,1)
    y_strt = np.array([crd[0][1] for crd in line_coords]).reshape(-1,1)
    x_ends = np.array([crd[1][0] for crd in line_coords]).reshape(-1,1)
    y_ends = np.array([crd[1][1] for crd in line_coords]).reshape(-1,1)
    # Matrices of the distances between (start,start), (end,end) and (start,end) of each line pair
    D_strt_strt = np.sqrt( (x_strt - x_strt.T)**2 + (y_strt - y_strt.T)**2 )
    D_ends_ends = np.sqrt( (x_ends - x_ends.T)**2 + (y_ends - y_ends.T)**2 )
    D_strt_ends = np.sqrt( (x_strt - x_ends.T)**2 + (y_strt - y_ends.T)**2 )
    # Fill diagonal (same line) with infinity
    np.fill_diagonal(D_strt_strt, np.inf)
    np.fill_diagonal(D_ends_ends, np.inf)
    np.fill_diagonal(D_strt_ends, np.inf)
    # Obtain the minimal distance and check if it is smaller than the maximal distance
    D_min = np.min(np.array([D_strt_strt,D_ends_ends,D_strt_ends]),axis=0)
    close_pairs = np.nonzero(D_min < max_distance)
    # Order as a list of tuples and remove doubles
    return list(dict.fromkeys([tuple(sorted((i,j))) for (i,j) in zip(*close_pairs)]))

def angle(lineA, lineB):
    # Find angle between 2 vectors
    vecA = np.array([lineA[0][0]-lineA[1][0], lineA[0][1]-lineA[1][1]]).reshape(-1,1)
    vecB = np.array([lineB[0][0]-lineB[1][0], lineB[0][1]-lineB[1][1]]).reshape(-1,1)
    vecA = vecA / np.linalg.norm(vecA)
    vecB = vecB / np.linalg.norm(vecB)
    return np.arccos(vecA.T @ vecB)[0][0]

def prune_lines(line_coords, ps, min_length, max_angle, max_distance):
    # Find close pairs that have an angle smaller than max_angle
    close_pairs = find_close_line_pairs(line_coords, max_distance=max_distance/ps)
    merge_pairs = {key: [] for key in range(len(line_coords))}
    for pair in close_pairs:
        lineA = line_coords[pair[0]]
        lineB = line_coords[pair[1]]
        ang = angle(lineA, lineB) * 180 / np.pi
        if (ang < max_angle) or ((180-ang)<max_angle):
            merge_pairs[pair[0]].append(pair[1])
            merge_pairs[pair[1]].append(pair[0])
    # Combine all pairs to merge into one subgraph
    G = nx.Graph(merge_pairs,directed=False)
    to_connect = list(nx.connected_components(G))
    # In each subgraph, keep the 2 coordinates that are furthest away from each other 
    pruned_line_coords = []
    for line_ids in to_connect:
        x_coords = np.array([line_coords[id][0][0] for id in line_ids] + [line_coords[id][1][0] for id in line_ids]).reshape(-1,1)
        y_coords = np.array([line_coords[id][0][1] for id in line_ids] + [line_coords[id][1][1] for id in line_ids]).reshape(-1,1)
        D = np.sqrt( (x_coords - x_coords.T)**2 + (y_coords - y_coords.T)**2 )
        if (D.max()>min_length/ps): # Remove lines that are too small
            ids_to_keep = np.unravel_index(D.argmax(), D.shape)
            pruned_line_coords.append(((x_coords.flatten()[ids_to_keep[0]],y_coords.flatten()[ids_to_keep[0]]),(x_coords.flatten()[ids_to_keep[1]],y_coords.flatten()[ids_to_keep[1]])))
    return pruned_line_coords

def draw_lines(line_coords,img_shape,cmap='gray',thickness=2):
    lines = np.zeros(img_shape,dtype='int')
    if cmap=='gray':
        for crd in line_coords:
            start,end = crd
            rr,cc = draw.line(start[1],start[0],end[1],end[0])
            lines[rr,cc] = 1
    elif cmap=='rgb':
        count = 0
        for crd in line_coords:
            count += 1
            start,end = crd
            rr,cc = draw.polygon([start[1]-thickness,end[1]-thickness,end[1]+thickness,start[1]+thickness],[start[0]-thickness,end[0]-thickness,end[0]+thickness,start[0]+thickness])
            lines[rr,cc] = count
    return lines

def format_coords_in_starfile(coords):
    out = '\n# version 30001\n\ndata_\n\nloop_ \n_rlnCoordinateX #1 \n_rlnCoordinateY #2 \n_rlnClassNumber #3 \n_rlnAnglePsi #4 \n_rlnAutopickFigureOfMerit #5 \n '
    for line in coords:
        out = out + f'{line[0][0]:.6f} {line[0][1]:.6f} 2 -999.00000 -999.00000 \n '    # Line starting coord
        out = out + f'{line[1][0]:.6f} {line[1][1]:.6f} 2 -999.00000 -999.00000 \n '    # Line ending coord
    out += '\n'
    return out

def prep_relative_output_dir(root,job_nr,rel_mrc_path,create_dir=False):
    out_dir = os.path.join('CustomDogPick',f'job{job_nr:03}')
    make_dir = False
    for dir in Path(rel_mrc_path).parts[:-1]:
        if make_dir:
            out_dir = os.path.join(out_dir,dir)
            if create_dir:
                Path(os.path.join(root,out_dir)).mkdir(parents=True, exist_ok=True) # Make path from 'job###' onwards
        if 'job' in dir:
            make_dir = True
    return out_dir

def write_coordinate_starfile(root,job_nr,rel_mrc_path,coords):   
    # Make output directories 
    rel_out_dir = prep_relative_output_dir(root,job_nr,rel_mrc_path,create_dir=True)
    # Write file
    out_fname = Path(os.path.split(rel_mrc_path)[-1]).stem + '_autopick.star'
    out_file = os.path.join(root,rel_out_dir,out_fname)
    with open(out_file,'w+') as f:
        f.write(format_coords_in_starfile(coords))
    return True

def write_autopick_starfile(root,job_nr,mrcfiles):    
    out = '\n# version 30001\n\ndata_coordinate_files\n\nloop_ \n_rlnMicrographName #1 \n_rlnMicrographCoordinates #2 \n'
    for rel_mrc_path in mrcfiles:
        rel_out_dir = prep_relative_output_dir(root,job_nr,rel_mrc_path,create_dir=False)
        out_fname = Path(os.path.split(rel_mrc_path)[-1]).stem + '_autopick.star'
        out += rel_mrc_path + ' ' + os.path.join(rel_out_dir,out_fname) + ' \n'
    out += ' \n'

    autopick_out_path = os.path.join(root,'CustomDogPick',f'job{job_nr:03}','customdogpick.star')
    with open(autopick_out_path,'w+') as f:
        f.write(out)
    return True


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    #-i DATABASE -u USERNAME -p PASSWORD -size 20
    parser.add_argument("-m", "--micrographs", help="Path to micrographs star")
    parser.add_argument("-r", "--root", help="Path to Relion root directory")
    parser.add_argument("-j", "--jobnr", help="Job number")
    args = parser.parse_args()
    path_to_micrographs_star = args.micrographs
    root = args.root
    job_nr = int(args.jobnr)

    with open(os.path.join(root,path_to_micrographs_star),'r') as f:
        lines = f.readlines()
    mrcfiles = [f.split()[0] for f in lines if '.mrc' in f]

    for rel_mrc_path in mrcfiles:
        print(f'Picking fibrils for mrc file {rel_mrc_path}...')
        mrc_path = os.path.join(root, rel_mrc_path)
        data,ps = read_data(mrc_path,rescale)
        dog = filters.difference_of_gaussians( normalize(data), low_sigma=dog_low_sigma/ps )
        skel = skeletonize(dog,ps,min_length,ridge_threshold)
        line_coords = detect_lines(skel,ps,hough_line_length,hough_line_gap)
        pruned_line_coords = prune_lines(line_coords, ps, min_length, max_angle, max_distance)
        write_coords = write_coordinate_starfile(root,job_nr,rel_mrc_path,pruned_line_coords)
    
    print(f'Writing output starfile...')
    write_star = write_autopick_starfile(root,job_nr,mrcfiles)
    print(f'Done!')