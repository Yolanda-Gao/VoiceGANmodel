## New Model Run Script Date: Oct 8th
## Author: Yang Gao

import os
import fnmatch
import argparse
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from dataset import *
from model_Adp import *
import scipy
import scipy.io as sio
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
import pdb
import librosa
from sklearn import preprocessing


parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--num_gpu', type=int, default=1) ## add num_gpu
parser.add_argument('--delta', type=str, default='true', help='Set to use or not use delta feature')
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
parser.add_argument('--task_name', type=str, default='spectrogram', help='Set data name')
parser.add_argument('--epoch_size', type=int, default=5, help='Set epoch size')
parser.add_argument('--batch_size', type=int, default=8, help='Set batch size')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate for optimizer')
parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--model_arch', type=str, default='spec_gan_n', help='choose among gan/recongan/discogan/spec_gan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN, spec_gan - My modified GAN model for speech.')
parser.add_argument('--image_size', type=int, default=256, help='Image size. 64 for every experiment in the paper')

parser.add_argument('--gan_curriculum', type=int, default=1000, help='Strong GAN loss for certain period at the beginning')
parser.add_argument('--starting_rate', type=float, default=0.01, help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
parser.add_argument('--default_rate', type=float, default=0.5, help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')

parser.add_argument('--n_test', type=int, default=20, help='Number of test data.')

parser.add_argument('--update_interval', type=int, default=10, help='') # origin 3

parser.add_argument('--log_interval', type=int, default=10, help='Print loss values every log_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=3000, help='Save test results every image_save_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=10000, help='Save models every model_save_interval iterations.')

def as_np(data):
    return data.cpu().data.numpy()

def get_data():
    if args.task_name == 'spectrogram':
       data_A = []
       directory = '/home/yang/DiscoGAN/datasets/spectrogram/man/train'
       for root, dirnames, filenames in os.walk(directory):
           for filename in fnmatch.filter(filenames, '*.npy'):
               # files.append(os.path.join(root, filename))
               data_A.append('./datasets/spectrogram/man/train/'+filename) 
         
       test_A = []
       directory = '/home/yang/DiscoGAN/datasets/spectrogram/man/val'
       for root, dirnames, filenames in os.walk(directory):
           for filename in fnmatch.filter(filenames, '*.npy'):
               # files.append(os.path.join(root, filename))
               test_A.append('./datasets/spectrogram/man/val/'+filename)
         
 
       data_B = []
       directory = '/home/yang/DiscoGAN/datasets/spectrogram/woman/train'
       for root, dirnames, filenames in os.walk(directory):
           for filename in fnmatch.filter(filenames, '*.npy'):
               # files.append(os.path.join(root, filename))
               data_B.append('./datasets/spectrogram/woman/train/'+filename)
 
       test_B = []
       directory = '/home/yang/DiscoGAN/datasets/spectrogram/woman/val'
       for root, dirnames, filenames in os.walk(directory):
           for filename in fnmatch.filter(filenames, '*.npy'):
               # files.append(os.path.join(root, filename))
               test_B.append('./datasets/spectrogram/woman/val/'+filename)

    return data_A, data_B, test_A, test_B


def get_fm_loss(real_feats, fake_feats, criterion):
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        # pdb.set_trace()
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        loss = criterion( l2, Variable( torch.ones( l2.size() ) ).cuda() )
        losses += loss

    return losses

## Change to 3 inputs 
def get_gan_loss(dis_real, dis_fake1, dis_fake2, criterion, cuda):
    labels_dis_real = Variable(torch.ones( [dis_real.size()[0], 1] ))
    labels_dis_fake1 = Variable(torch.zeros([dis_fake1.size()[0], 1] ))
    labels_dis_fake2 = Variable(torch.zeros([dis_fake2.size()[0], 1] ))
    labels_gen1 = Variable(torch.ones([dis_fake1.size()[0], 1]))
    labels_gen2 = Variable(torch.ones([dis_fake2.size()[0], 1]))

    if cuda:
        labels_dis_real = labels_dis_real.cuda()
        labels_dis_fake1 = labels_dis_fake1.cuda()
     	labels_dis_fake2 = labels_dis_fake2.cuda()
        labels_gen1 = labels_gen1.cuda()
	labels_gen2 = labels_gen2.cuda()

    dis_loss = criterion( dis_real, labels_dis_real ) * 0.4 + criterion( dis_fake1, labels_dis_fake1 ) * 0.3 + criterion( dis_fake2, labels_dis_fake2 ) * 0.3
    gen_loss = criterion( dis_fake1, labels_gen1 ) * 0.5 + criterion( dis_fake2, labels_gen2 ) * 0.5

    return dis_loss, gen_loss

## Use CrossEntropyLoss: target should be N
def get_stl_loss(A_stl, A1_stl, A2_stl, B_stl, B1_stl, B2_stl, criterion, cuda):
    # for nn.CrossEntropyLoss, the target is class index.
    labels_A = Variable(torch.ones( A_stl.size()[0] )) # NLL/CE target N not Nx1
    labels_A.data =  labels_A.data.type(torch.LongTensor)

    labels_A1 = Variable(torch.ones( A1_stl.size()[0] )) # NLL/CE target N not Nx1
    labels_A1.data =  labels_A1.data.type(torch.LongTensor)

    labels_A2 = Variable(torch.ones( A2_stl.size()[0] )) # NLL/CE target N not Nx1
    labels_A2.data =  labels_A2.data.type(torch.LongTensor)
 
    labels_B = Variable(torch.zeros(B_stl.size()[0] ))
    labels_B.data =  labels_B.data.type(torch.LongTensor)

    labels_B1 = Variable(torch.zeros(B1_stl.size()[0] ))
    labels_B1.data =  labels_B1.data.type(torch.LongTensor)

    labels_B2 = Variable(torch.zeros(B2_stl.size()[0] ))
    labels_B2.data =  labels_B2.data.type(torch.LongTensor)
   
    if cuda:
        labels_A = labels_A.cuda()
        labels_A1 = labels_A1.cuda()
        labels_A2 = labels_A2.cuda()
        labels_B = labels_B.cuda()
        labels_B1 = labels_B1.cuda()
        labels_B2 = labels_B2.cuda()

    A_stl = np.squeeze(A_stl)
    A1_stl = np.squeeze(A1_stl)
    A2_stl = np.squeeze(A2_stl)
    B_stl = np.squeeze(B_stl)
    B1_stl = np.squeeze(B1_stl)
    B2_stl = np.squeeze(B2_stl)

    stl_loss_A = criterion( A_stl, labels_A ) * 0.2 + criterion( A1_stl, labels_A1 ) * 0.15 + criterion( A2_stl, labels_A2 ) * 0.15
    stl_loss_B = criterion( B_stl, labels_B ) * 0.2 + criterion( B1_stl, labels_B1 ) * 0.15 + criterion( B2_stl, labels_B2 ) * 0.15
    stl_loss = stl_loss_A + stl_loss_B

    return stl_loss

def delta_regu(input_v, batch_size, criterion=nn.MSELoss()):
    losses = 0
    for i in range(batch_size):
        # pdb.set_trace()
        input_temp = np.squeeze(input_v.data[i,:,:,:])
        # no need to take mean among 3 channels since current input is 256x256 instead of 3x256x256
        # input_temp = np.mean(input_temp.cpu().numpy(), axis = 0)
        input_temp = input_temp.cpu().numpy()
        input_delta = np.absolute(librosa.feature.delta(input_temp))
        delta_loss = criterion(Variable((torch.from_numpy(input_delta)).type(torch.DoubleTensor)), Variable((torch.zeros([256,256])).type(torch.DoubleTensor)))
        # delta_loss = criterion((torch.from_numpy(input_delta)), Variable((torch.zeros([256,256]))))
        losses += delta_loss

    delta_losses = losses/batch_size

    return delta_losses.type(torch.cuda.FloatTensor)  

def normf(A):
    x = A.data.cpu().numpy()
    x_min = x.min(axis=(0, 1), keepdims=True)
    x_max = x.max(axis=(0, 1), keepdims=True)
    x = (x - x_min)/(x_max-x_min)
    x = Variable((torch.from_numpy(x)).type(torch.FloatTensor))
    return x


def main():

    global args, data_A
    args = parser.parse_args()


    cuda = args.cuda
    if cuda == 'true':
        cuda = True
    else:
        cuda = False

    task_name = args.task_name

    epoch_size = args.epoch_size
    batch_size = args.batch_size

    result_path = os.path.join( args.result_path, args.task_name )
    result_path = os.path.join( result_path, args.model_arch )

    model_path = os.path.join( args.model_path, args.task_name )
    model_path = os.path.join( model_path, args.model_arch )

    data_style_A, data_style_B, test_style_A, test_style_B = get_data()

    # pdb.set_trace() 

    if args.task_name == 'spectrogram':
       # test_A = read_spect( test_style_A, args.image_size )
       # test_B = read_spect( test_style_B, args.image_size )
       test_A = read_spect_matrix( test_style_A )
       test_B = read_spect_matrix( test_style_B )


    test_A_V = Variable( torch.FloatTensor( test_A ), volatile=True )
    test_B_V = Variable( torch.FloatTensor( test_B ), volatile=True )


    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    generator_A = Generator(args.num_gpu)
    generator_B = Generator(args.num_gpu)
    discriminator_A = Discriminator(args.num_gpu)
    discriminator_B = Discriminator(args.num_gpu)
    discriminator_S = StyleDiscriminator(args.num_gpu)

    if cuda:
        test_A_V = test_A_V.cuda()
        test_B_V = test_B_V.cuda()
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()
        discriminator_A = discriminator_A.cuda()
        discriminator_B = discriminator_B.cuda()
        discriminator_S = discriminator_S.cuda()

    if args.num_gpu > 1:
        # test_A_V = nn.DataParallel(test_A_V, device_ids = range(args.num_gpu))
        # test_B_V = nn.DataParallel(test_B_V, device_ids = range(args.num_gpu))
        generator_A = nn.DataParallel(generator_A, device_ids = range(args.num_gpu))
        generator_B = nn.DataParallel(generator_B, device_ids = range(args.num_gpu))
        discriminator_A = nn.DataParallel(discriminator_A, device_ids = range(args.num_gpu))
        discriminator_B = nn.DataParallel(discriminator_B, device_ids = range(args.num_gpu))
        discriminator_S = nn.DataParallel(discriminator_S, device_ids = range(args.num_gpu)) 

    data_size = min( len(data_style_A), len(data_style_B) )
    n_batches = ( data_size // batch_size )

    recon_criterion = nn.L1Loss() #MSELoss()
    gan_criterion = nn.BCELoss()
    feat_criterion = nn.HingeEmbeddingLoss()
    stl_criterion = nn.CrossEntropyLoss()

    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())
    stl_params =  discriminator_S.parameters() 

    optim_gen = optim.Adam( gen_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_dis = optim.Adam( dis_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_stl = optim.Adam( stl_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)

    iters = 0

    log_gen_loss = []
    log_dis_loss = []
    log_stl_loss = []

    for epoch in range(epoch_size):
        data_style_A, data_style_B = shuffle_data( data_style_A, data_style_B)

        widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval=n_batches, widgets=widgets)
        pbar.start()

        for i in range(n_batches):

            pbar.update(i)

            generator_A.zero_grad()
            generator_B.zero_grad()
            discriminator_A.zero_grad()
            discriminator_B.zero_grad()
            discriminator_S.zero_grad()

            A_path = data_style_A[ i * batch_size: (i+1) * batch_size ]
            B_path = data_style_B[ i * batch_size: (i+1) * batch_size ]

            if args.task_name =='spectrogram':
               A = read_spect_matrix( A_path )
               B = read_spect_matrix( B_path )

            A = Variable( torch.FloatTensor( A ) )
            B = Variable( torch.FloatTensor( B ) )

            if cuda:
                A = A.cuda()
                B = B.cuda()

            # A -> AB -> ABA
            # pdb.set_trace()
            A = A.unsqueeze(1)
            AB, AL_feats, LAB_feats = generator_B(A)
            ABA, ABL_feats, ABLA_feats = generator_A(AB)
            # B -> BA -> BAB
            B = B.unsqueeze(1)
            BA, BL_feats, LBA_feats = generator_A(B)
            BAB, BAL_feats, BALB_feats = generator_B(BA)
           
            # Reconstruction Loss
#            # adjust value to [0,1] before MSE            
#	    A_n = normf(A)
#           B_n = normf(B)  
#	    AB_n = normf(AB)
#	    BA_n = normf(BA)
#	    ABA_n = normf(ABA)
#	    BAB_n = normf(BAB) 
              

            # Target recon with input
            recon_loss_BA = recon_criterion( BA, B)
            recon_loss_AB = recon_criterion( AB, A)
            recon_loss_ABA = recon_criterion( ABA, A)
            recon_loss_BAB = recon_criterion( BAB, B)
            # pdb.set_trace()
#            recon_loss_A = 0.3*recon_loss_BA + 0.7*recon_loss_ABA
#            recon_loss_B = 0.3*recon_loss_AB + 0.7*recon_loss_BAB

            ### Should make it target same ###
	    recon_loss_A = 30 *recon_loss_AB + 70 *recon_loss_ABA
	    recon_loss_B = 30 *recon_loss_BA + 70 *recon_loss_BAB 

            # Real/Fake GAN Loss (A)
            A_dis = discriminator_A( A )
            BA_dis = discriminator_A( BA )
            ABA_dis = discriminator_A( ABA )
            # will be strange in one epoch
            dis_loss_A, gen_loss_A = get_gan_loss( A_dis, BA_dis, ABA_dis, gan_criterion, cuda )
            # pdb.set_trace()
            fm_loss_A1 = get_fm_loss(AL_feats, ABLA_feats, feat_criterion)
            fm_loss_A2 = get_fm_loss(LAB_feats, ABL_feats, feat_criterion)
	    fm_loss_A = fm_loss_A1 + fm_loss_A2         	

            # Real/Fake GAN Loss (B)
            B_dis = discriminator_B( B )
            AB_dis = discriminator_B( AB )
            BAB_dis = discriminator_B( BAB )

            dis_loss_B, gen_loss_B = get_gan_loss( B_dis, AB_dis, BAB_dis, gan_criterion, cuda )
            fm_loss_B1 = get_fm_loss( BL_feats, BALB_feats, feat_criterion )
            fm_loss_B2 = get_fm_loss( LBA_feats, BAL_feats, feat_criterion )
            fm_loss_B = fm_loss_B1 + fm_loss_B2

            # Style Discriminator Loss
            A_stl = discriminator_S(A)
            B_stl = discriminator_S(B)
            AB_stl = discriminator_S(AB) 
	    BA_stl = discriminator_S(BA)
	    ABA_stl = discriminator_S(ABA) 		
	    BAB_stl = discriminator_S(BAB)

            stl_loss = get_stl_loss(A_stl, BA_stl, ABA_stl, B_stl, AB_stl, BAB_stl, stl_criterion, cuda)

            # Delta regularizer
            log_delta_A = []
            log_delta_B = []
            BA_delta = delta_regu(BA, batch_size)
            AB_delta = delta_regu(AB, batch_size)
            ABA_delta = delta_regu(ABA, batch_size)
            BAB_delta = delta_regu(BAB, batch_size)

            delta_A = BA_delta + ABA_delta
            delta_B = AB_delta + BAB_delta

            # Total Loss

            if iters < args.gan_curriculum:
                rate = args.starting_rate
            else:
                rate = args.default_rate
            # origin 0.1 0.9
            # pdb.set_trace()
            gen_loss_A_total = (gen_loss_A) * (1.-rate) + (recon_loss_A*0.6 + fm_loss_A*0.3 + delta_A*0.1)*rate
            gen_loss_B_total = (gen_loss_B) * (1.-rate) + (recon_loss_B*0.6 + fm_loss_B*0.3 + delta_B*0.1)*rate

            if args.model_arch == 'discogan':
                gen_loss = gen_loss_A_total + gen_loss_B_total
                dis_loss = dis_loss_A + dis_loss_B
            elif args.model_arch == 'spec_gan_n':
                gen_loss = gen_loss_A_total + gen_loss_B_total
                dis_loss = dis_loss_A + dis_loss_B + stl_loss
            elif args.model_arch == 'recongan':
                gen_loss = gen_loss_A_total
                dis_loss = dis_loss_B
            elif args.model_arch == 'gan':
                gen_loss = (gen_loss_B*0.1 + fm_loss_B*0.9)
                dis_loss = dis_loss_B

            if iters % args.update_interval == 0:
                dis_loss.backward()
                optim_dis.step()
                optim_stl.step() 
            else:
                gen_loss.backward()
                optim_gen.step()

            if iters % args.log_interval == 0:
                print "---------------------"
                print "GEN Loss:", as_np(gen_loss_A.mean()), as_np(gen_loss_B.mean())
                print "Feature Matching Loss:", as_np(fm_loss_A.mean()), as_np(fm_loss_B.mean())
                print "RECON Loss:", as_np(recon_loss_A.mean()), as_np(recon_loss_B.mean())
                print "DIS Loss:", as_np(dis_loss_A.mean()), as_np(dis_loss_B.mean())
                print "Style Loss:", as_np(stl_loss.mean())
	        print "Delta Loss:", as_np(delta_A.mean()), as_np(delta_B.mean())

                log_gen_loss = np.concatenate([log_gen_loss, as_np(gen_loss.mean())]) 
                log_dis_loss = np.concatenate([log_dis_loss, as_np(dis_loss.mean())])
                log_stl_loss = np.concatenate([log_stl_loss, as_np(stl_loss.mean())])
                log_delta_A = np.concatenate([log_delta_A, as_np(delta_A.mean())])
                log_delta_B = np.concatenate([log_delta_B, as_np(delta_B.mean())])    

            if iters % args.image_save_interval == 0:
                # save test
                if test_A_V.data.shape[1] != 1:
                   test_A_V = test_A_V.unsqueeze(1)
                   test_B_V = test_B_V.unsqueeze(1)
                AB_test, testAL_feats, testLAB_feats = generator_B( test_A_V )
                BA_test, testBL_feats, testLBA_feats = generator_A( test_B_V )
                ABA_test, testABL_feats, testABLA_feats = generator_A( AB_test )
                BAB_test, testBAL_feats, testBALB_feats = generator_B( BA_test )

                n_testset = min( test_A_V.size()[0], test_B_V.size()[0] )

                subdir_path = os.path.join( result_path, str(iters / args.image_save_interval) )

                if os.path.exists( subdir_path ):
                    pass
                else:
                    os.makedirs( subdir_path )

                for im_idx in range( n_testset ):
                    # pdb.set_trace()                    
                    A_val = test_A_V[im_idx].cpu().data.numpy().transpose(1,2,0)# * 255.
                    B_val = test_B_V[im_idx].cpu().data.numpy().transpose(1,2,0)# * 255.

                    BA_val = BA_test[im_idx].cpu().data.numpy().transpose(1,2,0) # * 255.
                    ABA_val = ABA_test[im_idx].cpu().data.numpy().transpose(1,2,0)# * 255.
                    AB_val = AB_test[im_idx].cpu().data.numpy().transpose(1,2,0)# * 255.
                    BAB_val = BAB_test[im_idx].cpu().data.numpy().transpose(1,2,0)# * 255.

                    filename_prefix = os.path.join (subdir_path, str(im_idx))
                    sio.savemat(filename_prefix +'.A_val_0.mat', {'A_val_0':A_val})
                    sio.savemat(filename_prefix +'.B_val_0.mat', {'B_val_0':B_val})
                    sio.savemat(filename_prefix +'.BA_val_0.mat', {'BA_val_0':BA_val})
                    sio.savemat(filename_prefix +'.AB_val_0.mat', {'AB_val_0':AB_val})
                    sio.savemat(filename_prefix +'.ABA_val_0.mat', {'ABA_val_0':ABA_val})
                    sio.savemat(filename_prefix +'.BAB_val_0.mat', {'BAB_val_0':BAB_val})

                # save train

                n_trainset = batch_size

                subdir_path = os.path.join( result_path, str(iters / args.image_save_interval) )

                if os.path.exists( subdir_path ):
                    pass
                else:
                    os.makedirs( subdir_path )

                for im_idx in range( n_trainset ):

                    A_train = A[im_idx].cpu().data.numpy().transpose(1,2,0)# * 255.
                    B_train = B[im_idx].cpu().data.numpy().transpose(1,2,0)# * 255.

                    BA_train = BA[im_idx].cpu().data.numpy().transpose(1,2,0) # * 255.
                    ABA_train = ABA[im_idx].cpu().data.numpy().transpose(1,2,0)# * 255.
                    AB_train = AB[im_idx].cpu().data.numpy().transpose(1,2,0)# * 255.
                    BAB_train = BAB[im_idx].cpu().data.numpy().transpose(1,2,0)# * 255.

                    filename_prefix = os.path.join (subdir_path, str(im_idx))
                    sio.savemat(filename_prefix +'.A_train_0.mat', {'A_train_0':A_train})
                    sio.savemat(filename_prefix +'.B_train_0.mat', {'B_train_0':B_train})
                    sio.savemat(filename_prefix +'.BA_train_0.mat', {'BA_train_0':BA_train})
                    sio.savemat(filename_prefix +'.AB_train_0.mat', {'AB_train_0':AB_train})
                    sio.savemat(filename_prefix +'.ABA_train_0.mat', {'ABA_train_0':ABA_train})
                    sio.savemat(filename_prefix +'.BAB_train_0.mat', {'BAB_train_0':BAB_train})


            if iters % args.model_save_interval == 0:
                torch.save( generator_A, os.path.join(model_path, 'model_gen_A-' + str( iters / args.model_save_interval )))
                torch.save( generator_B, os.path.join(model_path, 'model_gen_B-' + str( iters / args.model_save_interval )))
                torch.save( discriminator_A, os.path.join(model_path, 'model_dis_A-' + str( iters / args.model_save_interval )))
                torch.save( discriminator_B, os.path.join(model_path, 'model_dis_B-' + str( iters / args.model_save_interval )))

            iters += 1


    sio.savemat('log_gen_loss.mat', {'log_gen_loss':log_gen_loss})
    sio.savemat('log_dis_loss.mat', {'log_dis_loss':log_dis_loss})
    sio.savemat('log_stl_loss.mat', {'log_stl_loss':log_stl_loss})
    sio.savemat('log_delta_A.mat', {'log_delta_A':log_delta_A})
    sio.savemat('log_delta_B.mat', {'log_delta_B':log_delta_B})


if __name__=="__main__":
    main()
