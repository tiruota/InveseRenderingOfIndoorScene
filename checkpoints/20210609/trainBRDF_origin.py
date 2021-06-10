from numpy.core.multiarray import concatenate
import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
import os
import models
import torchvision.utils as vutils
import utils
import dataloader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import turnErrorIntoNumpy, writeImageToFile

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default=None, help='path to input images')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic training setting
parser.add_argument('--nepoch0', type=int, default=14, help='the number of epochs for training')
parser.add_argument('--nepoch1', type=int, default=10, help='the number of epochs for training')

parser.add_argument('--batchSize0', type=int, default=1, help='input batch size')
parser.add_argument('--batchSize1', type=int, default=1, help='input batch size')

parser.add_argument('--imHeight0', type=int, default=720, help='the height / width of the input image to network')
parser.add_argument('--imWidth0', type=int, default=1280, help='the height / width of the input image to network')
parser.add_argument('--imHeight1', type=int, default=720, help='the height / width of the input image to network')
parser.add_argument('--imWidth1', type=int, default=1280, help='the height / width of the input image to network')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0, 1, 2], help='the gpus used for training network')
# Fine tune the network
parser.add_argument('--isFineTune', action='store_true', help='fine-tune the network')
parser.add_argument('--epochIdFineTune', type=int, default = 0, help='the training of epoch of the loaded model')
# The training weight
parser.add_argument('--albedoWeight', type=float, default=1.5, help='the weight for the diffuse component')
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for the diffuse component')
# parser.add_argument('--roughWeight', type=float, default=0.5, help='the weight for the roughness component')
parser.add_argument('--depthWeight', type=float, default=1.0, help='the weight for depth component')

parser.add_argument('--cascadeLevel', type=int, default=0, help='the casacade level')

# The detail network setting
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

albeW, normW = opt.albedoWeight, opt.normalWeight
# rougW = opt.roughWeight
deptW = opt.depthWeight

opt.nepoch = opt.nepoch0
opt.batchSize = opt.batchSize0
opt.imHeight, opt.imWidth = opt.imHeight0, opt.imWidth0

if opt.experiment is None:
    opt.experiment = 'check_w%d_h%d'%(opt.imWidth, opt.imHeight)
os.system('mkdir {0}'.format(opt.experiment))
os.system('cp *.py %s' % opt.experiment)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Initial Network
encoder = models.encoder()
albedoDecoder = models.decoder(mode=0)
normalDecoder = models.decoder(mode=1 )
# roughDecoder = models.decoder0(mode=2 )
depthDecoder = models.decoder(mode=4 )

lr_scale = 1
loss_L1 = nn.L1Loss().cuda()

#########################################
encoder = nn.DataParallel(encoder, device_ids = opt.deviceIds )
albedoDecoder = nn.DataParallel(albedoDecoder, device_ids = opt.deviceIds )
normalDecoder = nn.DataParallel(normalDecoder, device_ids = opt.deviceIds )
# roughDecoder = nn.DataParallel(roughDecoder, device_ids = opt.deviceIds )
depthDecoder = nn.DataParallel(depthDecoder, device_ids = opt.deviceIds )

##############  ######################
# Send things into GPU
if opt.cuda:
    encoder = encoder.cuda(opt.gpuId )
    albedoDecoder = albedoDecoder.cuda()
    normalDecoder = normalDecoder.cuda()
    # roughDecoder = roughDecoder.cuda()
    depthDecoder = depthDecoder.cuda()
####################################

####################################
# Optimizer
opEncoder = optim.Adam(encoder.parameters(), lr=2e-4 * lr_scale, betas=(0.5, 0.999) )
opAlbedo = optim.Adam(albedoDecoder.parameters(), lr=2e-4 * lr_scale, betas=(0.5, 0.999) )
opNormal = optim.Adam(normalDecoder.parameters(), lr=2e-4 * lr_scale, betas=(0.5, 0.999) )
# opRough = optim.Adam(roughDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
opDepth = optim.Adam(depthDecoder.parameters(), lr=2e-4 * lr_scale, betas=(0.5, 0.999) )
#####################################

brdfDataset = dataloader.BatchLoader(opt.dataRoot, imHeight=opt.imHeight, imWidth=opt.imWidth)
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize, num_workers = 8, shuffle = False)

j = 0
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
# roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

for epoch in list(range(opt.epochIdFineTune+1, opt.nepoch)):
    train_log = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
    for i, data_batch in enumerate(brdfLoader):
        j += 1
        # LOad data from cpu to gpu
        albedo_cpu = data_batch['albedo']
        albedo_batch = Variable(albedo_cpu).cuda()

        normal_cpu = data_batch['normal']
        normal_batch = Variable(normal_cpu).cuda()

        # rough_cpu = data_batch['rough']
        # rough_batch = Variable(rough_cpu ).cuda()

        depth_cpu = data_batch['depth']
        depth_batch = Variable(depth_cpu).cuda()

        segArea_cpu = data_batch['segArea']
        segEnv_cpu = data_batch['segEnv']
        segObj_cpu = data_batch['segObj']

        seg_cpu = torch.cat([segArea_cpu, segEnv_cpu, segObj_cpu], dim=1)
        seg_batch = Variable(seg_cpu).cuda()

        # segBRDF_batch = seg_batch[: 3:5, :, :]
        segBRDF_batch = seg_batch[:, 2:3, :, :]
        # print(segBRDF_batch.shape)
        # print(segBRDF_batch.shape)
        segAll_batch = seg_batch[:, 0:1, :, :] + seg_batch[:, 2:3, :, :]
        # print(segAll_batch.shape)
        # print(segAll_batch.shape)
        # segAll_batch = seg_batch[0] + seg_batch[1]

        # Load the image from cpu to gpu
        im_cpu = data_batch['rgb']
        im_batch = Variable(im_cpu).cuda()

        # Clear the gradient in optimizer
        opEncoder.zero_grad()
        opAlbedo.zero_grad()
        opNormal.zero_grad()
        # opRough.zero_grad()
        opDepth.zero_grad()

        ########################################################
        # Build the cascade network architecture #
        albedoPreds = []
        normalPreds = []
        # roughPreds = []
        depthPreds = []

        input_batch = im_batch

        # Initial Prediction
        x1, x2, x3, x4, x5, x6 = encoder(input_batch)
        albedo_pred = 0.5 * (albedoDecoder(im_batch, x1, x2, x3, x4, x5, x6) + 1)
        normal_pred = normalDecoder(im_batch, x1, x2, x3, x4, x5, x6)
        # roughPred = roughDecoder(imBatch, x1, x2, x3, x4, x5, x6)
        depth_pred = 0.5 * (depthDecoder(im_batch, x1, x2, x3, x4, x5, x6 ) + 1)
    
        albedo_batch = segBRDF_batch * albedo_batch
        albedo_pred = models.LSregress(albedo_pred * segBRDF_batch.expand_as(albedo_pred),
            albedo_batch * segBRDF_batch.expand_as(albedo_batch), albedo_pred)
        albedo_pred = torch.clamp(albedo_pred, 0, 1)
        
        depth_pred = models.LSregress(depth_pred * segAll_batch.expand_as(depth_pred),
            depth_batch * segAll_batch.expand_as(depth_batch), depth_pred)

        albedoPreds.append(albedo_pred)
        normalPreds.append(normal_pred)
        # roughPreds.append(roughPred )
        depthPreds.append(depth_pred)
        ########################################################

        # Compute the error
        albedoErrs = []
        normalErrs = []
        depthErrs = []
        # roughErrs = []

        pixelObjNum = (torch.sum(segBRDF_batch).cpu().data).item()
        pixelAllNum = (torch.sum(segAll_batch).cpu().data).item()
        for n in range(0, len(albedoPreds)):
            albedoErrs.append(torch.sum((albedoPreds[n] - albedo_batch)
                * (albedoPreds[n] - albedo_batch) * segBRDF_batch.expand_as(albedo_batch)) / pixelObjNum / 3.0)
        for n in range(0, len(normalPreds)):
            normalErrs.append(torch.sum((normalPreds[n] - normal_batch)
                * (normalPreds[n] - normal_batch) * segAll_batch.expand_as(normal_batch)) / pixelAllNum/ 3.0)
        for n in range(0, len(depthPreds)):
            depthErrs.append(torch.sum((torch.log(depthPreds[n]+1) - torch.log(depth_batch+1))
                * (torch.log(depthPreds[n]+1) - torch.log(depth_batch+1)) * segAll_batch.expand_as(depth_batch)) / pixelAllNum)

        # albedoErrs.append(loss_L1(albedo_pred, albedo_batch))
        # normalErrs.append(loss_L1(normal_pred, normal_batch))
        # depthErrs.append(loss_L1(depth_pred, depth_batch))

        # Back propagate the gradients
        # print(albedoErrs, normalErrs, depthErrs)
        totalErr = 4 * albeW * albedoErrs[-1] + normW * normalErrs[-1] + deptW * depthErrs[-1]
        totalErr.backward()

        # Update the network parameter
        opEncoder.step()
        opAlbedo.step()
        opNormal.step()
        # opRough.step()
        opDepth.step()

        utils.writeErrToScreen('albedo', albedoErrs, epoch, j)
        utils.writeErrToScreen('normal', normalErrs, epoch, j)
        # utils.writeErrToScreen('rough', roughErrs, epoch, j )
        utils.writeErrToScreen('depth', depthErrs, epoch, j)

        utils.writeErrToFile('albedo', albedoErrs, train_log, epoch, j)
        utils.writeErrToFile('normal', normalErrs, train_log, epoch, j)
        utils.writeErrToFile('depth', depthErrs, train_log, epoch, j)

        albedoErrsNpList = np.concatenate([albedoErrsNpList, utils.turnErrorIntoNumpy(albedoErrs)], axis=0)
        normalErrNplist = np.concatenate([normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs)], axis=0)
        depthErrNpList = np.concatenate([depthErrsNpList, utils.turnErrorIntoNumpy(depthErrs)], axis=0)

        if j < 1000:
            utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('normalAccu', np.mean(normalErrNplist[i:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), train_log, epoch, j)
            utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), train_log, epoch, j)
            # utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), train_log, epoch, j)

        else:
            utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            # utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), train_log, epoch, j)
            utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), train_log, epoch, j)
            # utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), train_log, epoch, j)

        if j == 1 or j%2000 == 0:
            # # Save the ground truth and the input
            # vutils.save_image(( (albedo_batch ) ** (1.0/2.2) ).data,
            #         '{0}/{1}_albedoGt.png'.format(opt.experiment, j) ,normalize=True)
            # vutils.save_image( (0.5*(normal_batch + 1) ).data,
            #         '{0}/{1}_normalGt.png'.format(opt.experiment, j), normalize=True)
            # # vutils.save_image( (0.5*(roughBatch + 1) ).data,
            # #         '{0}/{1}_roughGt.png'.format(opt.experiment, j) )
            # vutils.save_image( ( (im_batch)**(1.0/2.2) ).data,
            #         '{0}/{1}_im.png'.format(opt.experiment, j), normalize=True)
            # depthOut = 1 / torch.clamp(depth_batch + 1, 1e-6, 10) * segAll_batch.expand_as(depth_batch)
            # vutils.save_image( ( depthOut*segAll_batch.expand_as(depth_batch) ).data,
            #         '{0}/{1}_depthGt.png'.format(opt.experiment, j), normalize=True)

            utils.modelimg2cvimg(albedo_batch, '{0}/{1}_albedoGt.png'.format(opt.experiment, j))
            utils.modelimg2cvimg(normal_batch, '{0}/{1}_normalGt.png'.format(opt.experiment, j))
            utils.modelimg2cvimg(im_batch, '{0}/{1}_im.png'.format(opt.experiment, j))
            utils.modelimg2cvimg(depth_batch, '{0}/{1}_depthGt.png'.format(opt.experiment, j))


        # Save the predicted results
            for n in range(0, len(albedoPreds) ):
                vutils.save_image( ( (albedoPreds[n] ) ** (1.0/2.2) ).data,
                        '{0}/{1}_albedoPred_{2}.png'.format(opt.experiment, j, n) )
            for n in range(0, len(normalPreds) ):
                vutils.save_image( ( 0.5*(normalPreds[n] + 1) ).data,
                        '{0}/{1}_normalPred_{2}.png'.format(opt.experiment, j, n) )
            # for n in range(0, len(roughPreds) ):
            #     vutils.save_image( ( 0.5*(roughPreds[n] + 1) ).data,
            #             '{0}/{1}_roughPred_{2}.png'.format(opt.experiment, j, n) )
            for n in range(0, len(depthPreds) ):
                depthOut = 1 / torch.clamp(depthPreds[n] + 1, 1e-6, 10) * segAll_batch.expand_as(depthPreds[n])
                vutils.save_image( ( depthOut * segAll_batch.expand_as(depthPreds[n]) ).data,
                        '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, j, n) )

    train_log.close()

    # Update the training rate
    if (epoch + 1) % 10 == 0:
        for param_group in opEncoder.param_groups:
            param_group['lr'] /= 2
        for param_group in opAlbedo.param_groups:
            param_group['lr'] /= 2
        for param_group in opNormal.param_groups:
            param_group['lr'] /= 2
        # for param_group in opRough.param_groups:
        #     param_group['lr'] /= 2
        for param_group in opDepth.param_groups:
            param_group['lr'] /= 2
    
    # Save the error record
    np.save('{0}/albedoError_{1}.npy'.format(opt.experiment, epoch), albedoErrsNpList )
    np.save('{0}/normalError_{1}.npy'.format(opt.experiment, epoch), normalErrsNpList )
    # np.save('{0}/roughError_{1}.npy'.format(opt.experiment, epoch), roughErrsNpList )
    np.save('{0}/depthError_{1}.npy'.format(opt.experiment, epoch), depthErrsNpList )

    # save the models
    torch.save(encoder.module, '{0}/encoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(albedoDecoder.module, '{0}/albedo{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(normalDecoder.module, '{0}/normal{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    # torch.save(roughDecoder.module, '{0}/rough{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(depthDecoder.module, '{0}/depth{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )