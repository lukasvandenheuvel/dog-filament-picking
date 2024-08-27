import os
import yaml
import mrcfile
import numpy as np
import pandas as pd
import networkx as nx
from skimage import filters,morphology,transform,draw,io
from scipy.ndimage import convolve
from pathlib import Path
import multiprocessing
from functools import partial
import argparse

def list_mrc_files(path_to_micrographs_star):
    with open(os.path.join(root,path_to_micrographs_star),'r') as f:
        lines = f.readlines()
    nameline = [f.split()[1] for f in lines if '_rlnMicrographName' in f][0]
    col = int(nameline.split('#')[1])-1
    mrcfiles = [f.split()[col] for f in lines if '.mrc' in f]
    return mrcfiles

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
    product = vecA.T @ vecB
    # Make sure the product is in the range [-1,1]
    product = max( [min([product,1]), -1] )
    return np.arccos(product)[0][0]

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

def rescale_lines(coords,rescale):
    coords_rescaled = []
    for line in coords:
        coords_rescaled.append(((line[0][0]/rescale,line[0][1]/rescale),(line[1][0]/rescale,line[1][1]/rescale)))
    return coords_rescaled

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

def pick(rel_mrc_path,
         root,
         job_nr,
         rescale,
         dog_sigmas,
         min_length,
         ridge_threshold,
         hough_line_length,
         hough_line_gap,
         max_angle,
         max_distance,
         ellipse_kernel_size,
         ellipse_radius,
         ellipse_theta):
    print(f'Picking fibrils for mrc file {rel_mrc_path}...\n')

    # Apply DoG filters
    mrc_path = os.path.join(root, rel_mrc_path)
    data,ps = read_data(mrc_path,rescale)
    dog = filters.difference_of_gaussians( normalize(data), low_sigma=dog_sigmas[0]/ps )
    for sigma in dog_sigmas[1:]:
        dog = filters.difference_of_gaussians( normalize(dog), low_sigma=sigma/ps )

    # Create elliptical kernel
    ellipse = np.zeros((int(ellipse_kernel_size/ps),int(ellipse_kernel_size/ps)))
    rr,cc = draw.ellipse(int(ellipse_kernel_size/ps/2),int(ellipse_kernel_size/ps/2),int(ellipse_radius/ps/2),int(ellipse_kernel_size/ps/2))
    ellipse[rr,cc] = 1
    # Convolve with rotated ellipses
    line_coords = []
    for angle in np.linspace(0,360,ellipse_theta):
        ellipse_rot = transform.rotate(ellipse,angle=angle,resize=False)
        ellipse_rot = ellipse_rot / np.sum(ellipse_rot)
        cnv = convolve(dog,ellipse_rot)
        skel = skeletonize(cnv,ps,min_length,ridge_threshold)
        line_coords += detect_lines(skel,ps,hough_line_length,hough_line_gap)

    pruned_line_coords = prune_lines(line_coords, ps, min_length, max_angle, max_distance)
    rescaled_coords = rescale_lines(pruned_line_coords,rescale)
    write_coords = write_coordinate_starfile(root,job_nr,rel_mrc_path,rescaled_coords)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    #-i DATABASE -u USERNAME -p PASSWORD -size 20
    parser.add_argument("-y", "--yamlpath", help="Path to params yaml")
    parser.add_argument("-j", "--jobnr", help="Job number")
    parser.add_argument("-p", "--mpi", help="Number of MPIs")
    args = parser.parse_args()
    path_to_yaml = args.yamlpath
    job_nr = int(args.jobnr)
    mpi = int(args.mpi)

    with open(path_to_yaml, 'r') as file:
        params = yaml.safe_load(file)

    root = params['root']
    path_to_micrographs_star = params['path_to_micrographs_star']
    rescale             = params['rescale']
    sigma_view          = params['sigma_view']
    dog_sigmas          = params['dog_sigmas']
    ridge_sigmas        = params['ridge_sigmas']
    ridge_smoothing     = params['ridge_smoothing']
    ridge_threshold     = params['ridge_threshold']
    min_length          = params['min_length']
    hough_line_length   = params['hough_line_length']
    hough_line_gap      = params['hough_line_gap']
    max_angle           = params['max_angle']
    max_distance        = params['max_distance']
    ellipse_kernel_size = params['ellipse_kernel_size']
    ellipse_radius      = params['ellipse_radius']
    ellipse_theta       = params['ellipse_theta']

    mrcfiles = list_mrc_files(path_to_micrographs_star)
    pool = multiprocessing.Pool(processes=mpi)
    pool.map(partial(pick, 
                     root=root,
                     job_nr=job_nr,
                     rescale=rescale,
                     dog_sigmas=dog_sigmas,
                     min_length=min_length,
                     ridge_threshold=ridge_threshold,
                     hough_line_length=hough_line_length,
                     hough_line_gap=hough_line_gap,
                     max_angle=max_angle,
                     max_distance=max_distance,
                     ellipse_kernel_size=ellipse_kernel_size,
                     ellipse_radius=ellipse_radius,
                     ellipse_theta=ellipse_theta),
                     mrcfiles)

    # for rel_mrc_path in mrcfiles:
    #     print(f'Picking fibrils for mrc file {rel_mrc_path}...')
    #     mrc_path = os.path.join(root, rel_mrc_path)
    #     data,ps = read_data(mrc_path,rescale)
    #     dog = filters.difference_of_gaussians( normalize(data), low_sigma=dog_low_sigma/ps )
    #     skel = skeletonize(dog,ps,min_length,ridge_threshold)
    #     line_coords = detect_lines(skel,ps,hough_line_length,hough_line_gap)
    #     pruned_line_coords = prune_lines(line_coords, ps, min_length, max_angle, max_distance)
    #     write_coords = write_coordinate_starfile(root,job_nr,rel_mrc_path,pruned_line_coords)
    
    print(f'Writing output starfile...')
    write_star = write_autopick_starfile(root,job_nr,mrcfiles)
    print(f'Done!')