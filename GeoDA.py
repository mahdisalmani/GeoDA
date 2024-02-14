"""
Created on Thu Sep 26 01:34:01 2019

@author: AliRah
"""


import torch.nn as nn
import torchvision.datasets as dsets

import torchvision.transforms as transforms
import torchvision.models as torch_models
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from utils import    get_label
from utils import valid_bounds, clip_image_values
from PIL import Image
from torch.autograd import Variable
from numpy import linalg 
import math
from generate_2d_dct_basis import generate_2d_dct_basis
import time
import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_queries',
        type=int,
        default=1000,
        help='The max number of queries in model'
    )
    parser.add_argument(
        "--image_num",
        type=int,
        default=64,
        help="The index of the desired image"
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default="00000064",
        help="The index of the desired image"
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.6,
        help="mu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed"
    )
    return parser.parse_args()

def save_results(logs):
    numpy_results = np.full((2, args.max_queries), np.nan) 
    for i, my_intermediate in enumerate(logs):
        length = len(my_intermediate)
        for j in range(length):
            numpy_results[0][j] = logs[0][j]
            numpy_results[1][j] = logs[1][j]
    try:
        df = pd.read_csv(f'{args.seed}_results.csv', index_col=[0])
        df = df.append(pd.Series(numpy_results[0], index=df.columns[:len(numpy_results[0])]), ignore_index=True)
        df = df.append(pd.Series(numpy_results[1], index=df.columns[:len(numpy_results[1])]), ignore_index=True)
        df.to_csv(f'{args.seed}_results.csv')
    except: 
        pandas_results = pd.DataFrame(numpy_results)
        pandas_results.to_csv(f'{args.seed}_results.csv')
###############################################################
###############################################################

# Parameters 

args = get_args()

grad_estimator_batch_size = 40     # batch size for GeoDA


verbose_control = 'Yes'
#verbose_control = 'No'



Q_max = args.max_queries

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

sub_dim=75

tol = 0.0001
sigma = 0.0002
mu = args.mu


dist = 'l2'
# dist = 'linf'
# dist = 'l1'
# dist = 'linf'
search_space = 'sub'



image_iter = 0


image_num = args.image_num
inp = "./data/ILSVRC2012_val_" + str(args.image_name) + ".JPEG"



###############################################################
# Functions
###############################################################

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
    
    
def inv_tf(x, mean, std):

    for i in range(len(mean)):

        x[i] = np.multiply(x[i], std[i], dtype=np.float32)
        x[i] = np.add(x[i], mean[i], dtype=np.float32)

    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)

    return x

###############################################################


def is_adversarial(given_image, orig_label):
    
    predict_label = torch.argmax(net.forward(Variable(given_image, requires_grad=True)).data).item()

    return predict_label != orig_label

###############################################################
    
def find_random_adversarial(image, epsilon=1000):

    num_calls = 1
    
    step = 0.02
    perturbed = x_0
    
    while is_adversarial(perturbed, orig_label) == 0:
        
        pert = torch.randn([1,3,224,224])
        pert = pert.to(device)

        perturbed = image + num_calls*step* pert
        perturbed = clip_image_values(perturbed, lb, ub)
        perturbed = perturbed.to(device)
        num_calls += 1
        
    return perturbed, num_calls 

###############################################################            
            
def bin_search(x_0, x_random, tol):

    
    num_calls = 0
    adv = x_random
    cln = x_0
    
    while True:
        
        mid = (cln + adv) / 2.0
        num_calls += 1
        
        if is_adversarial(mid, orig_label):
            adv = mid
        else:
            cln = mid

        if torch.norm(adv-cln).cpu().numpy()<tol:
            break

    return adv, num_calls 

###############################################################

def black_grad_batch(x_boundary, q_max, sigma, random_noises, batch_size, original_label):
    
    grad_tmp = [] # estimated gradients in each estimate_batch
    z = []        # sign of grad_tmp
    outs = []
    num_batchs = math.ceil(q_max/batch_size)
    last_batch = q_max - (num_batchs-1)*batch_size
    EstNoise = SubNoise(batch_size, sub_basis_torch).cuda()
    all_noises = []
    for j in range(num_batchs):
        if j == num_batchs-1:
            EstNoise_last = SubNoise(last_batch, sub_basis_torch).cuda()
            current_batch = EstNoise_last()
            current_batch_np = current_batch.cpu().numpy()
            noisy_boundary = [x_boundary[0,:,:,:].cpu().numpy()]*last_batch +sigma*current_batch.cpu().numpy()

        else:
            current_batch = EstNoise()
            current_batch_np = current_batch.cpu().numpy()
            noisy_boundary = [x_boundary[0,:,:,:].cpu().numpy()]*batch_size +sigma*current_batch.cpu().numpy()
        
        all_noises.append(current_batch_np) 
        
        noisy_boundary_tensor = torch.tensor(noisy_boundary).to(device)
        
        predict_labels = torch.argmax(net.forward(noisy_boundary_tensor),1).cpu().numpy().astype(int)
        
        
        outs.append(predict_labels)
    all_noise = np.concatenate(all_noises, axis=0)
    outs = np.concatenate(outs, axis=0)
        

    for i, predict_label in enumerate(outs):
        if predict_label == original_label:
            z.append(1)
            grad_tmp.append(all_noise[i])
        else:
            z.append(-1)
            grad_tmp.append(-all_noise[i])
    
    grad = -(1/q_max)*sum(grad_tmp)
    
    grad_f = torch.tensor(grad).to(device)[None, :,:,:]

    return grad_f, sum(z)

###############################################################

def go_to_boundary(x_0, grad, x_b):

    epsilon = 5

    num_calls = 1
    perturbed = x_0 

    if dist == 'l1' or dist == 'l2':
        
        grads = grad


    if dist == 'linf':
        
        grads = torch.sign(grad)/torch.norm(grad)

        
        
    while is_adversarial(perturbed, orig_label) == 0:

        perturbed = x_0 + (num_calls*epsilon* grads[0])
        perturbed = clip_image_values(perturbed, lb, ub)

        num_calls += 1
        
        if num_calls > 100:
            print('falied ... ')
            break
        print
    return perturbed, num_calls, epsilon*num_calls 

###############################################################
def GeoDA(x_b, iteration, q_opt, q_num=0):
    
    
    norms = []
    grad = 0
    logs = np.zeros((2, iteration+1))
    logs[0][0] = 0
    logs[1][0] = linalg.norm(inv_tf(x_b.cpu().numpy()[0,:,:,:].squeeze(), mean, std)-image_fb)
    
    for i in range(iteration):
    
        t1 = time.time()
        random_vec_o = torch.randn(q_opt[i],3,224,224)

        grad_oi, ratios = black_grad_batch(x_b + sigma*(x_b-x_0)/torch.norm(x_b-x_0), q_opt[i], sigma, random_vec_o, grad_estimator_batch_size , orig_label)
        q_num = q_num + q_opt[i]
        grad = grad_oi + grad
        x_adv, qs, eps = go_to_boundary(x_0, grad, x_b)
        q_num = q_num + qs
        x_adv, bin_query = bin_search(x_0, x_adv, tol)


        q_num = q_num + bin_query
        print(q_num)

        x_b = x_adv
        
        t2 = time.time()
        x_adv_inv = inv_tf(x_adv.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
        
        if dist == 'l1' or dist == 'l2':
            dp = 'l2'
            norm_p = linalg.norm(x_adv_inv-image_fb)
            
            
        if dist == 'linf':
            dp = dist

            norm_p = np.max(abs(x_adv_inv-image_fb))
        
        if verbose_control == 'Yes':
            message = ' (took {:.5f} seconds)'.format(t2 - t1)
            print('iteration -> ' + str(i) + str(message) + '     -- ' + dp + ' norm is -> ' + str(norm_p))

        logs[0][i+1] = q_num
        logs[1][i+1] = norm_p
        
        
    x_adv = clip_image_values(x_adv, lb, ub)
        
    return x_adv, q_num, grad, logs



###############################################################

def opt_query_iteration(Nq, T, eta): 

  
    
    coefs=[eta**(-2*i/3) for i in range(0,T)]
    coefs[0] = 1*coefs[0]
    
    sum_coefs = sum(coefs)
    opt_q=[round(Nq*coefs[i]/sum_coefs) for i in range(0,T)]
    
    if opt_q[0]>80:
        T = T + 1
        opt_q, T = opt_query_iteration(Nq, T, eta)
    elif opt_q[0]<50:
        T = T - 1

        opt_q, T = opt_query_iteration(Nq, T, eta)

    return opt_q, T

def uni_query(Nq, T, eta): 

    opt_q=[round(Nq/T) for i in range(0,T)]
      
        
    return opt_q

###############################################################

def load_image(image, shape=(224, 224), data_format='channels_last'):

    assert len(shape) == 2
    assert data_format in ['channels_first', 'channels_last']

    
    image = image.resize(shape)
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    assert image.shape == shape + (3,)
    if data_format == 'channels_first':
        image = np.transpose(image, (2, 0, 1))
    return image
###############################################################

class SubNoise(nn.Module):
    """given subspace x and the number of noises, generate sub noises"""
    # x is the subspace basis
    def __init__(self, num_noises, x):
        self.num_noises = num_noises
        self.x = x
        super(SubNoise, self).__init__()

    def forward(self):
        

        r = torch.zeros([224 ** 2, 3*self.num_noises], dtype=torch.float32)
        noise = torch.randn([self.x.shape[1], 3*self.num_noises], dtype=torch.float32).cuda()
        sub_noise = torch.transpose(torch.mm(self.x, noise), 0, 1)
        r = sub_noise.view([ self.num_noises, 3, 224, 224])

        r_list = r
        return r_list
###############################################################
if search_space == 'sub':
    print('Check if DCT basis available ...')
    
    path = os.path.join(os.path.dirname(__file__), '2d_dct_basis_{}.npy'.format(sub_dim))
    if os.path.exists(path):
        print('Yes, we already have it ...')
        sub_basis = np.load('2d_dct_basis_{}.npy'.format(sub_dim)).astype(np.float32)
    else:
        print('Generating dct basis ......')
        sub_basis = generate_2d_dct_basis(sub_dim).astype(np.float32)
        print('Done!\n')


    estimate_batch = grad_estimator_batch_size
    sub_basis_torch = torch.from_numpy(sub_basis).cuda()
    EstNoise = SubNoise(estimate_batch, sub_basis_torch).cuda()
    random_vectors = EstNoise()
    random_vectors_np = random_vectors.cpu().numpy()

###############################################################
# Models

resnet50 = torch_models.resnet50(pretrained=True).eval()
if torch.cuda.is_available():
    resnet50 = resnet50.cuda()
meanfb = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
stdfb = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))




# Check for cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load a pretrained model
net = torch_models.resnet50(pretrained=True)
net = net.to(device)
net.eval()


####################################
## Load Image and Resize
#    

t11 = time.time()

im_orig = Image.open(inp)
im_sz = 224
im_orig = transforms.Compose([transforms.Resize((im_sz, im_sz))])(im_orig)


image_fb = load_image(im_orig, data_format='channels_last')
image_fb = image_fb / 255.  # because our model expects values in [0, 1]
 
image_fb_first = load_image(im_orig, data_format='channels_first')
image_fb_first = image_fb_first / 255.
   
# Bounds for Validity and Perceptibility
delta = 255
lb, ub = valid_bounds(im_orig, delta)
    
    # Transform data

im = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])(im_orig)


lb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(lb)
ub = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(ub)

im_deepfool = im.to(device)
lb = lb[None, :, :, :].to(device)
ub = ub[None, :, :, :].to(device)

x_0 = im[None, :, :, :].to(device)
x_0_np = x_0.cpu().numpy()

orig_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()
labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
str_label_orig = get_label(labels[int(orig_label)].split(',')[0])

ground_truth  = open(os.path.join('val.txt'), 'r').read().split('\n')

ground_name_label = ground_truth[image_num-1]
ground_label_split_all =  ground_name_label.split

ground_label_split =  ground_name_label.split()

ground_label =  ground_name_label.split()[1]
ground_label_int = int(ground_label)
    

str_label_ground = get_label(labels[int(ground_label)].split(',')[0])

    

   
          
if ground_label_int != int(orig_label):
    print('Already missclassified ... Lets try another one!')
    
else:    
    

    image_iter = image_iter + 1
      
    x0_inverse = inv_tf(x_0.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
    dif_norm = linalg.norm(x0_inverse-image_fb)
    
    
###################################

        
    x_random, query_random_1 = find_random_adversarial(x_0, epsilon=100)

    x_rnd_inverse = inv_tf(x_random.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
    norm_rnd_inv = linalg.norm(x_rnd_inverse-image_fb)
    
    is_adversarial(x_random, orig_label)
    
    label_random = torch.argmax(net.forward(Variable(x_random, requires_grad=True)).data).item()
    
    
    # Binary search
    
    x_boundary, query_binsearch_2 = bin_search(x_0, x_random, tol)
    x_b = x_boundary

    
    Norm_rnd = torch.norm(x_0-x_boundary)      
    x_bin_inverse = inv_tf(x_boundary.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
    norm_bin_rnd = linalg.norm(x_bin_inverse-image_fb)
    
    
    x_rnd_BA = np.swapaxes(x_bin_inverse, 0, 2)
    x_rnd_BA = np.swapaxes(x_rnd_BA, 1, 2)


    
    is_adversarial(x_boundary, orig_label)
    
    label_boundary = torch.argmax(net.forward(Variable(x_boundary, requires_grad=True)).data).item()
    
    query_rnd = query_binsearch_2

    ###################################
    # Run over iterations



    iteration = round(Q_max/500) 
    q_opt_it = int(Q_max  - (iteration)*25)
    q_opt_iter, iterate = opt_query_iteration(q_opt_it, iteration, mu )
    q_opt_it = int(Q_max  - (iterate)*25)
    q_opt_iter, iterate = opt_query_iteration(q_opt_it, iteration, mu )
    print('#################################################################')
    print('Start: The GeoDA will be run for:' + ' Iterations = ' + str(iterate) + ', Query = ' + str(Q_max) + ', Norm = ' + str(dist)+ ', Space = ' + str(search_space) )
    print('#################################################################')


    t3 = time.time()
    x_adv, query_o, gradient, logs = GeoDA(x_b, iterate, q_opt_iter, query_rnd)
    t4 = time.time()
    message = ' took {:.5f} seconds'.format(t4 - t3)
    qmessage = ' with query = ' + str(query_o)

    x_opt_inverse = inv_tf(x_adv.cpu().numpy()[0,:,:,:].squeeze(), mean, std)
    norm_inv_opt = linalg.norm(x_opt_inverse-image_fb)
    save_results(logs)
               
    print('#################################################################')
    print('End: The GeoDA algorithm' + message + qmessage )
    print('#################################################################')