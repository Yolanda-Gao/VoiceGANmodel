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
# from model import *
from model_styleD import *
import scipy
import scipy.io as sio
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
import pdb
import librosa

# from .monitor import Monitor

parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--num_gpu', type=int, default=1) ## add num_gpu
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
parser.add_argument('--task_name', type=str, default='spectrogram', help='Set data name')
parser.add_argument('--epoch_size', type=int, default=2, help='Set epoch size')
parser.add_argument('--batch_size', type=int, default=16, help='Set batch size')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate for optimizer')
parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
parser.add_argument('--image_size', type=int, default=256, help='Image size. 64 for every experiment in the paper')
parser.add_argument('--n_test', type=int, default=20 , help='Number of test data.')
parser.add_argument('--log_interval', type=int, default=1 , help='Print loss values every log_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=1, help='Save test results every image_save_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=100, help='Save models every model_save_interval iterations.')

def as_np(data):
    return data.cpu().data.numpy()

def get_data():
    # celebA / edges2shoes / edges2handbags / ...
    if args.task_name == 'spectrogram':
         data_A = []
         directory = '/home/yang/DiscoGAN/datasets/spectrogram_shuffle/man/train'
         for root, dirnames, filenames in os.walk(directory):
             for filename in fnmatch.filter(filenames, '*.npy'): # was '*.png'
                 # files.append(os.path.join(root, filename))
                 data_A.append('./datasets/spectrogram_shuffle/man/train/'+filename) 
         # print(data_A)
         
         test_A = []
         # directory = '/home/yang/DiscoGAN/datasets/spectrogram/man/val'
         # directory = '/home/yang/DiscoGAN/results/spectrogram/discogan_trainsaved_4/24'
         directory = '/home/yang/DiscoGAN/results/spectrogram/spec_gan_sepoch10/14'
         for root, dirnames, filenames in os.walk(directory):
             # for filename in fnmatch.filter(filenames, '*.npy'):
             for filename in fnmatch.filter(filenames, '*.BA_val_0.mat'):
                 # files.append(os.path.join(root, filename))
                 test_A.append('./results/spectrogram/spec_gan_sepoch10/14/'+filename)
                 # test_A.append('./datasets/spectrogram/man/val/'+filename)
 
         data_B = []
         directory = '/home/yang/DiscoGAN/datasets/spectrogram_shuffle/woman/train'
         for root, dirnames, filenames in os.walk(directory):
             for filename in fnmatch.filter(filenames, '*.npy'):
                 # files.append(os.path.join(root, filename))
                 data_B.append('./datasets/spectrogram_shuffle/woman/train/'+filename)
 
         test_B = []
         # directory = '/home/yang/DiscoGAN/results/spectrogram/discogan_trainsaved_4/24'
         # directory = '/home/yang/DiscoGAN/datasets/spectrogram/woman/val'
         directory = '/home/yang/DiscoGAN/results/spectrogram/spec_gan_sepoch10/19'
         for root, dirnames, filenames in os.walk(directory):
             # for filename in fnmatch.filter(filenames, '*.npy'):
             for filename in fnmatch.filter(filenames, '*.AB_val_0.mat'):
                 # files.append(os.path.join(root, filename))
                 test_B.append('./results/spectrogram/spec_gan_sepoch10/19/'+filename)
                 # test_B.append('./datasets/spectrogram/woman/val/'+filename)
         # print(test_B)

    return data_A, data_B, test_A, test_B

def get_dis_loss(pre_A, pre_B, criterion, cuda):
    # for nn.CrossEntropyLoss, the target is class index.
    labels_A = Variable(torch.ones( pre_A.size()[0] )) # NLL/CE target N not Nx1
    labels_A.data =  labels_A.data.type(torch.LongTensor)
    # labels_A.data[:,1] = torch.zeros([pre_A.size()[0], 1]) # column index as 0,1 

    labels_B = Variable(torch.zeros(pre_B.size()[0] )) 
    # labels_B.data[:,1] = torch.zeros([pre_B.size()[0], 1])
    labels_B.data =  labels_B.data.type(torch.LongTensor)
 
    if cuda:
        labels_A = labels_A.cuda()
        labels_B = labels_B.cuda()
    
    # pdb.set_trace()
    pre_A = np.squeeze(pre_A)
    pre_B = np.squeeze(pre_B)
    # pdb.set_trace()
    
    dis_loss = criterion( pre_A, labels_A ) * 0.5 + criterion( pre_B, labels_B ) * 0.5

    ## calculate the prediction accuracy
    # pdb.set_trace()
    batch_size_A = pre_A.size(0)
    predictions_A = pre_A.max(1)[1].type_as(labels_A)
    correct_A = predictions_A.eq(labels_A)
    if not hasattr(correct_A, 'sum'):
       correct_A = correct_A.cpu()
    # pdb.set_trace()
    correct_A = torch.sum(correct_A.float()) #.sum()
    # pdb.set_trace()
    accuracy_A =  100. * correct_A.float() / batch_size_A

    batch_size_B = pre_B.size(0)
    predictions_B = pre_B.max(1)[1].type_as(labels_B)
    correct_B = predictions_B.eq(labels_B)
    if not hasattr(correct_B, 'sum'):
       correct_B = correct_B.cpu()
    correct_B = correct_B.float().sum()
    accuracy_B =  100. * correct_B.float() / batch_size_B

    return dis_loss, accuracy_A, accuracy_B

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

    if args.task_name == 'spectrogram':
        # pdb.set_trace()
        test_A = read_spect_matrix( test_style_A )
        test_B = read_spect_matrix( test_style_B )

    test_A_V = Variable( torch.FloatTensor( test_A ), volatile=True )
    test_B_V = Variable( torch.FloatTensor( test_B ), volatile=True )

    if cuda:
	test_A_V = test_A_V.cuda()
	test_B_V = test_B_V.cuda()


    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    discriminator_AB = Discriminator(args.num_gpu)

    if cuda:
        discriminator_AB = discriminator_AB.cuda()

    if args.num_gpu > 1:
        # test_A_V = nn.DataParallel(test_A_V, device_ids = range(args.num_gpu))
        # test_B_V = nn.DataParallel(test_B_V, device_ids = range(args.num_gpu))
        discriminator_AB = nn.DataParallel(discriminator_AB, device_ids = range(args.num_gpu))

    data_size = min( len(data_style_A), len(data_style_B) )
    n_batches = ( data_size // batch_size )

    dis_criterion = nn.CrossEntropyLoss()   # nn.BCELoss()

    dis_params = discriminator_AB.parameters()

    optim_dis = optim.Adam( dis_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)

    iters = 0

    log_DIS_ab = [] 
    log_DIS_ab_test = []

    for epoch in range(epoch_size):
        data_style_A, data_style_B = shuffle_data( data_style_A, data_style_B)

        widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval=n_batches, widgets=widgets)
        pbar.start()

        for i in range(n_batches):

            pbar.update(i)

            discriminator_AB.zero_grad()

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

            # pdb.set_trace()
            A = A.unsqueeze(1)
            B = B.unsqueeze(1)
            dis_A = discriminator_AB( A )
            dis_B = discriminator_AB( B )

            dis_loss, acc_A, acc_B = get_dis_loss( dis_A, dis_B, dis_criterion, cuda )

            dis_loss.backward()
            optim_dis.step()

            if iters % args.log_interval == 0:
                print "---------------------"
                print "DIS Loss:", as_np(dis_loss.mean())
                print "TRAIN A/B Accuracy:", as_np(acc_A), as_np(acc_B)
                log_DIS_ab = np.concatenate([log_DIS_ab, as_np(dis_loss.mean())])


            if iters % args.image_save_interval == 0:
                # test data
                if iters == -1: # 0 before
                   test_A_V = test_A_V.unsqueeze(1)
                   test_B_V = test_B_V.unsqueeze(1)
                
                dis_A_test = discriminator_AB( test_A_V )                 
                dis_B_test = discriminator_AB( test_B_V )
                dis_loss_test, T_acc_A, T_acc_B = get_dis_loss( dis_A_test, dis_B_test, dis_criterion, cuda ) 
                print "---------------------"
                print "TEST DIS Loss:", as_np(dis_loss_test.mean())
                print "TEST accuracy A/B:", as_np(T_acc_A), as_np(T_acc_B)

                # print "TEST A/B Mean Pred:", as_np(dis_A_test.mean()), as_np(dis_B_test.mean())

                log_DIS_ab_test = np.concatenate([log_DIS_ab_test, as_np(dis_loss_test.mean())])

            if iters % args.model_save_interval == 0:
                torch.save( discriminator_AB, os.path.join(model_path, 'model_dis_AB-' + str( iters / args.model_save_interval )))

            iters += 1


    sio.savemat('log_DIS_ab.mat', {'log_DIS_ab':log_DIS_ab})
    sio.savemat('log_DIS_ab_test.mat', {'log_DIS_ab_test':log_DIS_ab_test})


if __name__=="__main__":
    main()
