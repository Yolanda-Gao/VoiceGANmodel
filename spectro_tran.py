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
from model_delta import *
import scipy
import scipy.io as sio
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
import pdb
import librosa

parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--num_gpu', type=int, default=1) ## add num_gpu
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
parser.add_argument('--task_name', type=str, default='spectrogram', help='Set data name')
parser.add_argument('--epoch_size', type=int, default=5000, help='Set epoch size')
parser.add_argument('--batch_size', type=int, default=16, help='Set batch size')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate for optimizer')
parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
parser.add_argument('--image_size', type=int, default=256, help='Image size. 64 for every experiment in the paper')

parser.add_argument('--gan_curriculum', type=int, default=1000, help='Strong GAN loss for certain period at the beginning')
parser.add_argument('--starting_rate', type=float, default=0.01, help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
parser.add_argument('--default_rate', type=float, default=0.5, help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')

parser.add_argument('--style_A', type=str, default=None, help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
parser.add_argument('--style_B', type=str, default=None, help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
parser.add_argument('--constraint', type=str, default=None, help='Constraint for celebA dataset. Only images satisfying this constraint is used. For example, if --constraint=Male, and --constraint_type=1, only male images are used for both style/domain.')
parser.add_argument('--constraint_type', type=str, default=None, help='Used along with --constraint. If --constraint_type=1, only images satisfying the constraint are used. If --constraint_type=-1, only images not satisfying the constraint are used.')
parser.add_argument('--n_test', type=int, default=20 , help='Number of test data.')

parser.add_argument('--update_interval', type=int, default=10, help='') # origin 3

parser.add_argument('--log_interval', type=int, default=10, help='Print loss values every log_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=1000, help='Save test results every image_save_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=10000, help='Save models every model_save_interval iterations.')

def as_np(data):
    return data.cpu().data.numpy()

def get_data():
    # celebA / edges2shoes / edges2handbags / ...
    if args.task_name == 'facescrub':
        data_A, data_B = get_facescrub_files(test=False, n_test=args.n_test)
        test_A, test_B = get_facescrub_files(test=True, n_test=args.n_test)

    elif args.task_name == 'celebA':
        data_A, data_B = get_celebA_files(style_A=args.style_A, style_B=args.style_B, constraint=args.constraint, constraint_type=args.constraint_type, test=False, n_test=args.n_test)
        test_A, test_B = get_celebA_files(style_A=args.style_A, style_B=args.style_B, constraint=args.constraint, constraint_type=args.constraint_type, test=True, n_test=args.n_test)

    elif args.task_name == 'edges2shoes':
        data_A, data_B = get_edge2photo_files( item='edges2shoes', test=False )
        test_A, test_B = get_edge2photo_files( item='edges2shoes', test=True )

    elif args.task_name == 'edges2handbags':
        data_A, data_B = get_edge2photo_files( item='edges2handbags', test=False )
        test_A, test_B = get_edge2photo_files( item='edges2handbags', test=True )

    elif args.task_name == 'handbags2shoes':
        data_A_1, data_A_2 = get_edge2photo_files( item='edges2handbags', test=False )
        test_A_1, test_A_2 = get_edge2photo_files( item='edges2handbags', test=True )

        data_A = np.hstack( [data_A_1, data_A_2] )
        test_A = np.hstack( [test_A_1, test_A_2] )

        data_B_1, data_B_2 = get_edge2photo_files( item='edges2shoes', test=False )
        test_B_1, test_B_2 = get_edge2photo_files( item='edges2shoes', test=True )

        data_B = np.hstack( [data_B_1, data_B_2] )
        test_B = np.hstack( [test_B_1, test_B_2] )

    elif args.task_name == 'spectrogram':
         data_A = []
         directory = '/home/yang/DiscoGAN/datasets/spectrogram_matrix/man/train'
         for root, dirnames, filenames in os.walk(directory):
             for filename in fnmatch.filter(filenames, '*.npy'): # was '*.png'
                 # files.append(os.path.join(root, filename))
                 data_A.append('./datasets/spectrogram_matrix/man/train/'+filename) 
         # print(data_A)
         
         test_A = []
         directory = '/home/yang/DiscoGAN/datasets/spectrogram_matrix/man/val'
         for root, dirnames, filenames in os.walk(directory):
             for filename in fnmatch.filter(filenames, '*.npy'):
                 # files.append(os.path.join(root, filename))
                 test_A.append('./datasets/spectrogram_matrix/man/val/'+filename)
         
 
         data_B = []
         directory = '/home/yang/DiscoGAN/datasets/spectrogram_matrix/woman/train'
         for root, dirnames, filenames in os.walk(directory):
             for filename in fnmatch.filter(filenames, '*.npy'):
                 # files.append(os.path.join(root, filename))
                 data_B.append('./datasets/spectrogram_matrix/woman/train/'+filename)
 
         test_B = []
         directory = '/home/yang/DiscoGAN/datasets/spectrogram_matrix/woman/val'
         for root, dirnames, filenames in os.walk(directory):
             for filename in fnmatch.filter(filenames, '*.npy'):
                 # files.append(os.path.join(root, filename))
                 test_B.append('./datasets/spectrogram_matrix/woman/val/'+filename)
         # print(test_B)

    return data_A, data_B, test_A, test_B

def get_fm_loss(real_feats, fake_feats, criterion):
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        pdb.set_trace()
        loss = criterion( l2, Variable( torch.ones( l2.size() ) ).cuda() )
        losses += loss

    return losses

def get_gan_loss(dis_real, dis_fake, criterion, cuda):
    labels_dis_real = Variable(torch.ones( [dis_real.size()[0], 1] ))
    labels_dis_fake = Variable(torch.zeros([dis_fake.size()[0], 1] ))
    labels_gen = Variable(torch.ones([dis_fake.size()[0], 1]))

    if cuda:
        labels_dis_real = labels_dis_real.cuda()
        labels_dis_fake = labels_dis_fake.cuda()
        labels_gen = labels_gen.cuda()

    dis_loss = criterion( dis_real, labels_dis_real ) * 0.5 + criterion( dis_fake, labels_dis_fake ) * 0.5
    gen_loss = criterion( dis_fake, labels_gen )

    return dis_loss, gen_loss

def delta_regu(input_v, batch_size, criterion=nn.MSELoss()):
    losses = 0
    for i in range(batch_size):
        input_temp = np.squeeze(input_v.data[i,:,:,:])
        input_temp = np.mean(input_temp.cpu().numpy(), axis = 0)
        input_delta = np.absolute(librosa.feature.delta(input_temp))
        delta_loss = criterion(Variable((torch.from_numpy(input_delta)).type(torch.DoubleTensor)), Variable((torch.zeros([256,256])).type(torch.DoubleTensor)))
        # delta_loss = criterion((torch.from_numpy(input_delta)), Variable((torch.zeros([256,256]))))
        losses += delta_loss

    delta_losses = losses/batch_size

    return delta_losses.type(torch.cuda.FloatTensor)  

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
    if args.style_A:
        result_path = os.path.join( result_path, args.style_A )
    result_path = os.path.join( result_path, args.model_arch )

    model_path = os.path.join( args.model_path, args.task_name )
    if args.style_A:
        model_path = os.path.join( model_path, args.style_A )
    model_path = os.path.join( model_path, args.model_arch )

    data_style_A, data_style_B, test_style_A, test_style_B = get_data()

    if args.task_name.startswith('edges2'):
        test_A = read_images( test_style_A, 'A', args.image_size )
        test_B = read_images( test_style_B, 'B', args.image_size )

    elif args.task_name == 'handbags2shoes' or args.task_name == 'shoes2handbags':
        test_A = read_images( test_style_A, 'B', args.image_size )
        test_B = read_images( test_style_B, 'B', args.image_size )

    elif args.task_name == 'spectrogram':
        # test_A = read_spect( test_style_A, args.image_size )
        # test_B = read_spect( test_style_B, args.image_size )
        test_A = read_spect_matrix( test_style_A )
        test_B = read_spect_matrix( test_style_B )


    else:
        test_A = read_images( test_style_A, None, args.image_size )
        test_B = read_images( test_style_B, None, args.image_size )


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

    if cuda:
        test_A_V = test_A_V.cuda()
        test_B_V = test_B_V.cuda()
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()
        discriminator_A = discriminator_A.cuda()
        discriminator_B = discriminator_B.cuda()

    if args.num_gpu > 1:
        # test_A_V = nn.DataParallel(test_A_V, device_ids = range(args.num_gpu))
        # test_B_V = nn.DataParallel(test_B_V, device_ids = range(args.num_gpu))
        generator_A = nn.DataParallel(generator_A, device_ids = range(args.num_gpu))
        generator_B = nn.DataParallel(generator_B, device_ids = range(args.num_gpu))
        discriminator_A = nn.DataParallel(discriminator_A, device_ids = range(args.num_gpu))
        discriminator_B = nn.DataParallel(discriminator_B, device_ids = range(args.num_gpu))

  

    data_size = min( len(data_style_A), len(data_style_B) )
    n_batches = ( data_size // batch_size )

    recon_criterion = nn.MSELoss()
    gan_criterion = nn.BCELoss()
    feat_criterion = nn.HingeEmbeddingLoss()

    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())

    optim_gen = optim.Adam( gen_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_dis = optim.Adam( dis_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)

    iters = 0

    gen_loss_total = []
    dis_loss_total = []

    log_GEN_a = [] 
    log_FL_a = [] 
    log_RECON_a = [] 
    log_DIS_a = [] 

    log_GEN_b = [] 
    log_FL_b = [] 
    log_RECON_b = [] 
    log_DIS_b = [] 

    log_gen_loss = []
    log_dis_loss = []

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

            A_path = data_style_A[ i * batch_size: (i+1) * batch_size ]
            B_path = data_style_B[ i * batch_size: (i+1) * batch_size ]

            if args.task_name.startswith( 'edges2' ):
                A = read_images( A_path, 'A', args.image_size )
                B = read_images( B_path, 'B', args.image_size )
            elif args.task_name =='handbags2shoes' or args.task_name == 'shoes2handbags':
                A = read_images( A_path, 'B', args.image_size )
                B = read_images( B_path, 'B', args.image_size )
            elif args.task_name =='spectrogram':
#                A = read_spect( A_path, args.image_size )
#                B = read_spect( B_path, args.image_size )
                 A = read_spect_matrix( A_path )
                 B = read_spect_matrix( B_path )
            else:
                A = read_images( A_path, None, args.image_size )
                B = read_images( B_path, None, args.image_size )

            A = Variable( torch.FloatTensor( A ) )
            B = Variable( torch.FloatTensor( B ) )

            if cuda:
                A = A.cuda()
                B = B.cuda()

            # A = A.unsqueeze(1)
            # B = B.unsqueeze(2) 
            AB = generator_B(A)
            BA = generator_A(B)

            ABA = generator_A(AB)
            BAB = generator_B(BA)
      
           
            # Reconstruction Loss
            recon_loss_A = recon_criterion( ABA, A )
            recon_loss_B = recon_criterion( BAB, B )

            # Real/Fake GAN Loss (A)
            A_dis_real, A_feats_real = discriminator_A( A )
            A_dis_fake, A_feats_fake = discriminator_A( BA )

            dis_loss_A, gen_loss_A = get_gan_loss( A_dis_real, A_dis_fake, gan_criterion, cuda )
            fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, feat_criterion)

            # Real/Fake GAN Loss (B)
            B_dis_real, B_feats_real = discriminator_B( B )
            B_dis_fake, B_feats_fake = discriminator_B( AB )

            dis_loss_B, gen_loss_B = get_gan_loss( B_dis_real, B_dis_fake, gan_criterion, cuda )
            fm_loss_B = get_fm_loss( B_feats_real, B_feats_fake, feat_criterion )

            # Delta regularizer
            delta_A = []
            delta_B = []
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
            gen_loss_A_total = (gen_loss_B*0.3 + fm_loss_B*0.7) * (1.-rate) + (recon_loss_A*0.6 + delta_A*0.4)*rate
            gen_loss_B_total = (gen_loss_A*0.3 + fm_loss_A*0.7) * (1.-rate) + (recon_loss_B*0.6 + delta_B*0.4)*rate

            if args.model_arch == 'discogan':
                gen_loss = gen_loss_A_total + gen_loss_B_total
                dis_loss = dis_loss_A + dis_loss_B
            elif args.model_arch == 'recongan':
                gen_loss = gen_loss_A_total
                dis_loss = dis_loss_B
            elif args.model_arch == 'gan':
                gen_loss = (gen_loss_B*0.1 + fm_loss_B*0.9)
                dis_loss = dis_loss_B

            if iters % args.update_interval == 0:
                dis_loss.backward()
                optim_dis.step()
            else:
                gen_loss.backward()
                optim_gen.step()

            if iters % args.log_interval == 0:
                print "---------------------"
                print "GEN Loss:", as_np(gen_loss_A.mean()), as_np(gen_loss_B.mean())
                print "Feature Matching Loss:", as_np(fm_loss_A.mean()), as_np(fm_loss_B.mean())
                print "RECON Loss:", as_np(recon_loss_A.mean()), as_np(recon_loss_B.mean())
                print "DIS Loss:", as_np(dis_loss_A.mean()), as_np(dis_loss_B.mean())
                log_GEN_a = np.concatenate([log_GEN_a, as_np(gen_loss_A.mean())])
                log_FL_a = np.concatenate([log_FL_a, as_np(fm_loss_A.mean())])
                log_RECON_a = np.concatenate([log_RECON_a, as_np(recon_loss_A.mean())])
                log_DIS_a = np.concatenate([log_DIS_a, as_np(dis_loss_A.mean())])

                log_GEN_b = np.concatenate([log_GEN_b, as_np(gen_loss_B.mean())])
                log_FL_b = np.concatenate([log_FL_b, as_np(fm_loss_B.mean())])
                log_RECON_b = np.concatenate([log_RECON_b, as_np(recon_loss_B.mean())])
                log_DIS_b = np.concatenate([log_DIS_b, as_np(dis_loss_B.mean())])                

                log_gen_loss = np.concatenate([log_gen_loss, as_np(gen_loss.mean())]) 
                log_dis_loss = np.concatenate([log_dis_loss, as_np(dis_loss.mean())])


            if iters % args.image_save_interval == 0:
                # save test
                AB_test = generator_B( test_A_V )
                BA_test = generator_A( test_B_V )
                ABA_test = generator_A( AB_test )
                BAB_test = generator_B( BA_test )

                n_testset = min( test_A_V.size()[0], test_B_V.size()[0] )

                subdir_path = os.path.join( result_path, str(iters / args.image_save_interval) )

                if os.path.exists( subdir_path ):
                    pass
                else:
                    os.makedirs( subdir_path )

                for im_idx in range( n_testset ):
                    
                    A_val = test_A_V[im_idx].cpu().data.numpy().transpose(1,2,0)# * 255.
                    B_val = test_B_V[im_idx].cpu().data.numpy().transpose(1,2,0)# * 255.
                    # print(A_val)

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

    sio.savemat('log_GEN_a.mat', {'log_GEN_a':log_GEN_a})
    sio.savemat('log_FL_a.mat', {'log_FL_a':log_FL_a})
    sio.savemat('log_RECON_a.mat', {'log_RECON_a':log_GEN_a})
    sio.savemat('log_DIS_a.mat', {'log_DIS_a':log_GEN_a})

    sio.savemat('log_GEN_b.mat', {'log_GEN_b':log_GEN_b})
    sio.savemat('log_FL_b.mat', {'log_FL_b':log_FL_b})
    sio.savemat('log_RECON_b.mat', {'log_RECON_b':log_GEN_b})
    sio.savemat('log_DIS_b.mat', {'log_DIS_b':log_GEN_b})

    sio.savemat('log_gen_loss.mat', {'log_gen_loss':log_gen_loss})
    sio.savemat('log_dis_loss.mat', {'log_dis_loss':log_dis_loss})


if __name__=="__main__":
    main()
