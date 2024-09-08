import os,yaml,sys
import mrcfile
import pandas as pd
from skimage import io,filters,draw,transform
import numpy as np
import napari
from custom_dog_picking_par import read_data,normalize,rescale_lines,write_coordinate_starfile,write_autopick_starfile
from scipy.ndimage import convolve
import multiprocessing
from functools import partial
import argparse

def within_frame(x,max_size):
    return max([0,min([x,max_size-1])])

def read_star_in_pandas(path_to_particles_star):
    # Read star
    with open(path_to_particles_star,'r') as f:
        lines = f.readlines()
    # Obtain header info
    num_header = 0
    columns = {}
    for i,l in enumerate(lines):
        if '_rln' in l:
            num_header = max([num_header,i])
            colsplit = l.split(' #')
            col_num = int(colsplit[1])-1
            columns[col_num] = colsplit[0]
    df = pd.read_csv(path_to_particles_star,skiprows=num_header+1,header=None,delimiter=' ').rename(columns=columns)
    return df

def pick_local(rel_mrc_path,
               root,
               job_nr,
               ptcls,
               rescale,
               sigma_background,
               dog_sigmas,
               ellipse_kernel_size,
               ellipse_radius,
               ellipse_theta,
               fibril_length
               ):
    print(f'Picking local fibrils for mrc file {rel_mrc_path}...\n')

    # Obtain particles on this micrograph and their coords
    ptcls_on_mg = ptcls[(ptcls._rlnMicrographName==rel_mrc_path)]
    points = [[ptcls_on_mg.iloc[i].rescaledY,ptcls_on_mg.iloc[i].rescaledX] for i in range(len(ptcls_on_mg))]

    # Preprocess micrograph
    mrc_path = os.path.join(root, rel_mrc_path)
    mg,mg_rescale,ps = read_data(mrc_path,rescale)
    height,width = mg_rescale.shape
    background_subtract = (mg_rescale - filters.gaussian( mg_rescale, sigma=sigma_background/ps ))
    dog = filters.difference_of_gaussians( normalize(background_subtract), low_sigma=dog_sigmas[0]/ps )
    for sigma in dog_sigmas[1:]:
        dog = filters.difference_of_gaussians( normalize(dog), low_sigma=sigma/ps )
    
    # Create elliptical kernel
    ellipse = np.zeros((int(ellipse_kernel_size/ps),int(ellipse_kernel_size/ps)))
    rr,cc = draw.ellipse(int(ellipse_kernel_size/ps/2),int(ellipse_kernel_size/ps/2),int(ellipse_radius/ps/2),int(ellipse_kernel_size/ps/2))
    ellipse[rr,cc] = 1
    # Match with rotated ellipses
    angles = np.arange(0,180+sys.float_info.epsilon,ellipse_theta)
    fibril_score = np.zeros((len(points),len(angles)))
    for a,angle in enumerate(angles):
        # Rotate the ellipse
        ellipse_rot = transform.rotate(ellipse,angle=angle,resize=False)==0 # Invert the ellipse to detect dark ridges
        ellipse_rot = ellipse_rot - np.mean(ellipse_rot)
        cnv = convolve(dog,ellipse_rot) # template matching by convolution
        # Store the values of template-matched image per point in an array
        for p,point in enumerate(points):
            fibril_score[p,a] = cnv[point[0],point[1]]
    
    # Obtain the orientations for which template matching was maximal
    orientations = angles[np.argmax(fibril_score,axis=1)] * np.pi / 180
    # Store the lines corresponding to the maximal angle in a list
    helix_lines = []
    for p,point in enumerate(points):
        dx = int(fibril_length * np.cos(orientations[p]) / 2 / ps)
        dy = -int(fibril_length * np.sin(orientations[p]) / 2 / ps) # y is negative because of the coordinate system in images
        helix_lines.append( (
            (within_frame(point[0]-dy,height),within_frame(point[1]-dx,width)), # Starting coordinate of line
            (within_frame(point[0]+dy,height),within_frame(point[1]+dx,width))  # Ending coordinate of line
        ))
    # Rescale and save
    rescaled_coords = rescale_lines(helix_lines,rescale)
    write_coords = write_coordinate_starfile(root,job_nr,rel_mrc_path,rescaled_coords,dir_name='LocalHelixPick',job_in_dirname=False)

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

    # Obtain params
    root                = params['root']
    path_to_ptcl_star   = params['path_to_star']
    rescale             = params['rescale']
    sigma_background    = params['sigma_background']
    dog_sigmas          = params['dog_sigmas']
    ellipse_kernel_size = params['ellipse_kernel_size']
    ellipse_radius      = params['ellipse_radius']
    ellipse_theta       = params['ellipse_theta']
    fibril_length       = params['fibril_length']

    # Read particle star file
    ptcls = read_star_in_pandas(path_to_ptcl_star)
    ptcls['rescaledX'] = (rescale * ptcls._rlnCoordinateX).astype('int')
    ptcls['rescaledY'] = (rescale * ptcls._rlnCoordinateY).astype('int')

    # Unique micrographs
    unique_mgs = list(ptcls._rlnMicrographName.unique())

    # Run with parallel pool
    pool = multiprocessing.Pool(processes=mpi)
    pool.map(partial(pick_local, 
                     root=root,
                     job_nr=job_nr,
                     ptcls=ptcls,
                     rescale=rescale,
                     sigma_background=sigma_background,
                     dog_sigmas=dog_sigmas,
                     ellipse_kernel_size=ellipse_kernel_size,
                     ellipse_radius=ellipse_radius,
                     ellipse_theta=ellipse_theta,
                     fibril_length=fibril_length),
                     unique_mgs)

    print(f'Writing output starfile...')
    write_star = write_autopick_starfile(root,job_nr,unique_mgs,dir_name='LocalHelixPick',starfile_name='localhelixpick.star',job_in_dirname=False)
    print(f'Done!')