"""
Adapted from https://github.com/thusiyuan/cooperative_scene_parsing/blob/master/utils/sunrgbd_utils.py
"""

from __future__ import print_function
import numpy as np
from PIL import Image
import cv2
import os
import os.path as osp
import torch
from torch.autograd import Variable
import h5py

def modelimg2cvimg(tensor, fname):
    img  = tensor[0,:,:,:].data.cpu().numpy()
    img = np.clip(img, -1, 1)
    img = (128 * img.transpose([1, 2, 0]) + 128).astype(np.uint8)
    if img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)
    cv2.imwrite(fname, img)

def srgb2rgb(srgb ):
    ret = np.zeros_like(srgb )
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power( (srgb[idx1] + 0.055) / 1.055, 2.4 )
    return ret

def writeErrToScreen(errorName, errorArr, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName), end=' ')
    for n in range(0, len(errorArr) ):
        print('%.6f' % errorArr[n].data.item(), end = ' ')
    print('.')

def writeCoefToScreen(coefName, coef, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(coefName), end=' ')
    coefNp = coef.cpu().data.numpy()
    for n in range(0, len(coefNp) ):
        print('%.6f' % coefNp[n], end = ' ')
    print('.')

def writeNpErrToScreen(errorName, errorArr, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName), end=' ')
    for n in range(0, len(errorArr) ):
        print('%.6f' % errorArr[n], end = ' ')
    print('.')


def writeErrToFile(errorName, errorArr, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}:'% (epoch, j) ).format(errorName) )
    for n in range(0, len(errorArr) ):
        fileOut.write('%.6f ' % errorArr[n].data.item() )
    fileOut.write('.\n')

def writeCoefToFile(coefName, coef, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}: ' % (epoch, j) ).format(coefName) )
    coefNp = coef.cpu().data.numpy()
    for n in range(0, len(coefNp) ):
        fileOut.write('%.6f ' % coefNp[n] )
    fileOut.write('.\n')

def writeNpErrToFile(errorName, errorArr, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName) )
    for n in range(0, len(errorArr) ):
        fileOut.write('%.6f ' % errorArr[n] )
    fileOut.write('.\n')

def turnErrorIntoNumpy(errorArr):
    errorNp = []
    for n in range(0, len(errorArr) ):
        errorNp.append(errorArr[n].data.item() )
    return np.array(errorNp)[np.newaxis, :]



def writeImageToFile(imgBatch, nameBatch, isGama = False):
    batchSize = imgBatch.size(0)
    for n in range(0, batchSize):
        img = imgBatch[n, :, :, :].data.cpu().numpy()
        img = np.clip(img, 0, 1)
        if isGama:
            img = np.power(img, 1.0/2.2)
        img = (255 *img.transpose([1, 2, 0] ) ).astype(np.uint8)
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img )
        img.save(nameBatch[n] )


def writeNumpyToFile(imBatch, nameBatch):
    batchSize = imBatch.size(0)
    for n in range(0, batchSize):
        im = imBatch[n, :, :, :].data.cpu().numpy()
        np.save(nameBatch[n], im)

def writeNumpzToFile(imBatch, nameBatch):
    batchSize = imBatch.size(0)
    for n in range(0, batchSize):
        im = imBatch[n, :, :, :].data.cpu().numpy()
        np.savez_compressed(nameBatch[n], data = im)


def writeH5ToFile(imBatch, nameBatch):
    batchSize = imBatch.size(0)
    assert(batchSize == len(nameBatch ) )
    for n in range(0, batchSize):
        im = imBatch[n, :, :, :].data.cpu().numpy()
        hf = h5py.File(nameBatch[n], 'w')
        hf.create_dataset('data', data=im, compression = 'lzf')
        hf.close()


def writeEnvToFile(envmaps, envId, envName, nrows=12, ncols=8, envHeight=8, envWidth=16, gap=1):
    envmap = envmaps[envId, :, :, :, :, :].data.cpu().numpy()
    envmap = np.transpose(envmap, [1, 2, 3, 4, 0] )
    envRow, envCol = envmap.shape[0], envmap.shape[1]

    interY = int(envRow / nrows )
    interX = int(envCol / ncols )

    lnrows = len(np.arange(0, envRow, interY) )
    lncols = len(np.arange(0, envCol, interX) )

    lenvHeight = lnrows * (envHeight + gap) + gap
    lenvWidth = lncols * (envWidth + gap) + gap

    envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
    for r in range(0, envRow, interY ):
        for c in range(0, envCol, interX ):
            rId = int(r / interY )
            cId = int(c / interX )

            rs = rId * (envHeight + gap )
            cs = cId * (envWidth + gap )
            envmapLarge[rs : rs + envHeight, cs : cs + envWidth, :] = envmap[r, c, :, :, :]

    envmapLarge = np.clip(envmapLarge, 0, 1)
    envmapLarge = (255 * (envmapLarge ** (1.0/2.2) ) ).astype(np.uint8 )
    cv2.imwrite(envName, envmapLarge[:, :, ::-1] )

def writeNumpyEnvToFile(envmap, envName, nrows=12, ncols=8, envHeight=8, envWidth=16, gap=1):
    envRow, envCol = envmap.shape[0], envmap.shape[1]

    interY = int(envRow / nrows )
    interX = int(envCol / ncols )

    lnrows = len(np.arange(0, envRow, interY) )
    lncols = len(np.arange(0, envCol, interX) )

    lenvHeight = lnrows * (envHeight + gap) + gap
    lenvWidth = lncols * (envWidth + gap) + gap

    envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
    for r in range(0, envRow, interY ):
        for c in range(0, envCol, interX ):
            rId = int(r / interY )
            cId = int(c / interX )

            rs = rId * (envHeight + gap )
            cs = cId * (envWidth + gap )
            envmapLarge[rs : rs + envHeight, cs : cs + envWidth, :] = envmap[r, c, :, :, :]

    envmapLarge = np.clip(envmapLarge, 0, 1)
    envmapLarge = (255 * envmapLarge ** (1.0/2.2) ).astype(np.uint8 )
    cv2.imwrite(envName, envmapLarge[:, :, ::-1] )

def predToShading(pred, envWidth = 32, envHeight = 16, SGNum = 12 ):

    Az = ( (np.arange(envWidth) + 0.5) / envWidth - 0.5 )* 2 * np.pi
    El = ( (np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0
    Az, El = np.meshgrid(Az, El)
    Az = Az[np.newaxis, :, :]
    El = El[np.newaxis, :, :]
    lx = np.sin(El) * np.cos(Az)
    ly = np.sin(El) * np.sin(Az)
    lz = np.cos(El)
    ls = np.concatenate((lx, ly, lz), axis = 0)
    ls = ls[np.newaxis, :, np.newaxis, np.newaxis, :, :]
    envWeight = np.cos(El) * np.sin(El )
    envWeight = envWeight[np.newaxis, np.newaxis, np.newaxis, :, :]

    envRow, envCol = pred.shape[2], pred.shape[3]
    pred = pred.squeeze(0)
    axisOrig = pred[0:3*SGNum, :, :]
    lambOrig = pred[3*SGNum : 4*SGNum, :, :]
    weightOrig = pred[4*SGNum : 7*SGNum, :, :]

    weight = weightOrig.reshape([SGNum, 3, envRow, envCol] ) * 0.999
    weight = np.tan(np.pi / 2.0 * weight )
    weight = weight[:, :, :, :, np.newaxis, np.newaxis ]

    axisDir = axisOrig.reshape([SGNum, 3, envRow, envCol] )
    axisDir = axisDir[:, :, :, :, np.newaxis, np.newaxis]

    lamb = lambOrig.reshape([SGNum, 1, envRow, envCol] ) * 0.999
    lamb = np.tan(np.pi / 2.0 * lamb )
    lamb = lamb[:, :, :, :, np.newaxis, np.newaxis]

    mi = lamb * (np.sum(axisDir * ls, axis=1)[:, np.newaxis, :, :, :, :] - 1)
    envmaps = np.sum(weight * np.exp(mi ), axis=0)

    shading = (envmaps * envWeight ).reshape([3, envRow, envCol, -1] )
    shading = np.sum(shading, axis = 3)
    shading = np.maximum(shading, 0.0)

    return shading

def normalize(vector):
    return vector / np.linalg.norm(vector)

def parse_camera_info(camera_info, height, width):
    """ extract intrinsic and extrinsic matrix
    """
    lookat = normalize(camera_info[3:6])
    up = normalize(camera_info[6:9])

    W = lookat
    U = np.cross(W, up)
    V = -np.cross(W, U)

    rot = np.vstack((U, V, W))
    trans = camera_info[:3]

    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.diag([1, 1, 1])

    K[0, 2] = width / 2
    K[1, 2] = height / 2

    K[0, 0] = K[0, 2] / np.tan(xfov)
    K[1, 1] = K[1, 2] / np.tan(yfov)

    return rot, trans, K


def flip_towards_viewer(normals, points):
    points = points / np.linalg.norm(points)
    proj = points.dot(normals[:2, :].T)
    flip = np.where(proj > 0)
    normals[flip, :] = -normals[flip, :]
    return normals


def get_corners_of_bb3d(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    # order the basis
    index = np.argsort(np.abs(basis[:, 0]))[::-1]
    # the case that two same value appear the same time
    if index[2] != 2:
        index[1:] = index[1:][::-1]
    basis = basis[index, :]
    coeffs = coeffs[index]
    # Now, we know the basis vectors are orders X, Y, Z. Next, flip the basis vectors towards the viewer
    basis = flip_towards_viewer(basis, centroid)
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners


def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]

    corners = corners + np.tile(centroid, (8, 1))
    return corners


def project_3d_points_to_2d(points3d, R_ex, K):
    """
    Project 3d points from camera-centered coordinate to 2D image plane
    Parameters
    ----------
    points3d: numpy array
        3d location of point
    R_ex: numpy array
        extrinsic camera parameter
    K: numpy array
        intrinsic camera parameter
    Returns
    -------
    points2d: numpy array
        2d location of the point
    """
    points3d = R_ex.dot(points3d.T).T
    x3 = points3d[:, 0]
    y3 = -points3d[:, 1]
    z3 = np.abs(points3d[:, 2])
    xx = x3 * K[0, 0] / z3 + K[0, 2]
    yy = y3 * K[1, 1] / z3 + K[1, 2]
    points2d = np.vstack((xx, yy))
    return points2d


def project_struct_bdb_to_2d(basis, coeffs, center, R_ex, K):
    """
    Project 3d bounding box to 2d bounding box
    Parameters
    ----------
    basis, coeffs, center, R_ex, K
        : K is the intrinsic camera parameter matrix
        : Rtilt is the extrinsic camera parameter matrix in right hand coordinates
    Returns
    -------
    bdb2d: dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    """
    corners3d = get_corners_of_bb3d(basis, coeffs, center)
    corners = project_3d_points_to_2d(corners3d, R_ex, K)
    bdb2d = dict()
    bdb2d['x1'] = int(max(np.min(corners[0, :]), 1))  # x1
    bdb2d['y1'] = int(max(np.min(corners[1, :]), 1))  # y1
    bdb2d['x2'] = int(min(np.max(corners[0, :]), 2*K[0, 2]))  # x2
    bdb2d['y2'] = int(min(np.max(corners[1, :]), 2*K[1, 2]))  # y2
    # if not check_bdb(bdb2d, 2*K[0, 2], 2*K[1, 2]):
    #     bdb2d = None
    return bdb2d

def visualize_depth_image(depth_image, fname, path="./"):
    depth_image =  int(depth_image/(depth_image.max()/255.0))
    cv2.imwrite(os.path.join(path, fname), depth_image)