import os
import math
import time
import yaml
import torch
import torchvision
import numpy as np
from scipy.integrate import quad
import torch.nn.init as init
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from LoadData import LoadHistologyData, transformation_list_training, transformation_list_no_train


# To find the elapsed time for execution #
class TimeElapsed:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


# Different types of weight initilization method #
def weights_init(init_type='gaussian'):
    def init_fun(m):
        class_name = m.__class__.__name__
        if (class_name.find('Conv') == 0 or class_name.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


# To load the scheduler #
def get_scheduler(optimizer, hyper_parameters, iterations=-1):
    if 'lr_policy' not in hyper_parameters or hyper_parameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyper_parameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyper_parameters['step_size'],
                                        gamma=hyper_parameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyper_parameters['lr_policy'])
    return scheduler


# To get path information of the data #
def get_model_list(dir_name, key):
    if os.path.exists(dir_name) is False:
        return None
    gen_models = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if
                  os.path.isfile(os.path.join(dir_name, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


# To get the data loader for training the model #
def get_data_loader_folder(input_folder, batch_size, train, new_size=256, height=256, width=256, num_workers=4):
    transform_list = transformation_list_training(new_size=new_size, height=height, width=width)
    dataset = LoadHistologyData(root_img=input_folder, transform=transform_list, train_flag=train)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


def get_data_loader_folder_no_train(input_folder, batch_size, new_size=256, height=256, width=256, num_workers=4):
    train_flag = False
    transform_list = transformation_list_no_train(new_size=new_size, height=height, width=width)
    dataset = LoadHistologyData(root_img=input_folder, transform=transform_list, train_flag=train_flag)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return loader
    

# To load the data loader using data path for training #
def get_loader(config, train_data=False, validation_data=False, template_data=False, source_data=False):
    if train_data:
        train_flag = True
        data_route = config['train_route']
    if validation_data:
    	train_flag = True
    	data_route = config['validation_route']
    if template_data:
    	train_flag = False
    	data_route = config['template_route']
    if source_data:
    	train_flag = False
    	data_route = config['source_route']

    batch_size = config['batch_size']
    num_workers = config['num_workers']
    new_size = config['new_size']
    height = config['crop_image_height']
    width = config['crop_image_width']
    if train_flag:
    	loader = get_data_loader_folder(input_folder=data_route, batch_size=batch_size, train=train_flag, new_size=new_size,
                                          height=height, width=width, num_workers=num_workers)
    else:
    	loader = get_data_loader_folder_no_train(input_folder=data_route, batch_size=batch_size, new_size=new_size,
                                                      num_workers=num_workers)
    return loader


# To prepare folders for saving intermediate images & checkpoints #
def prepare_folder(output_directory):
    image_directory = os.path.join(output_directory, 'Images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'Checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


# Writing loss values #
def write_loss(iteration, trainer, train_writer):
    members = [attr for attr in dir(trainer) if not callable(getattr(trainer, attr)) and not attr.startswith("__") and
               ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iteration + 1)


# Load the configuration file to read the details regarding hyper-parameters #
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.full_load(stream) 


# For writing images #
def write_image(image_output, display_image_num, file_name):
    # expand gray-scale images to 2 channels
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_output]
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = torchvision.utils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    torchvision.utils.save_image(image_grid, file_name, nrow=1)


# Writing images in desired directory #
def write_image_output(image_output, display_image_num, image_directory, postfix):
    n = len(image_output)
    write_image(image_output[0:n], display_image_num, '%s/gen_k_value_%s.ppm' % (image_directory, postfix))


# Compute KL divergence between Gaussian Distributions #
def compute_KLDiv_Gaussians(mean1, mean2, variance1, variance2):
    log_term = math.log(variance2/variance1)
    mean_variance_term = ((variance1 - variance1) + (mean1 - mean2)**2)/(2*variance2)
    KLdiv = log_term + mean_variance_term
    return KLdiv


# Compute KL divergence value between two multivariate Gaussians where first one is standard normal #
def compute_KLDiv_multivariate_Gaussians_standard_normal(mean_estimated, covariance_estimated):
    dim = mean_estimated.shape[0]
    inverse_covariance = np.linalg.inv(covariance_estimated)
    print("Covariance value: ", np.linalg.det(covariance_estimated))

    # KL divergence is made o three terms: here, the first probability distribution is standard normal (mean=0, covariance=identity matrix) #
    trace_term = np.trace(inverse_covariance)
    det_term = np.log(np.linalg.det(covariance_estimated))
    quad_term = (mean_estimated.T @ inverse_covariance @ mean_estimated)

    kldiv_value = 0.5 * (trace_term + det_term + quad_term - dim)
    return kldiv_value


# Approximation of KL Divergence between mixtures of Gaussian Distributions #
# pis: mixing factors corresponding to mixture model f, omegas: mixing factors corresponding to mixing model g#
def compute_KLDiv_mixture_Gaussians(num_components, means_f, means_g, variances_f, variances_g, pis, omegas):
    KLdiv_variational = 0.0
    for i in range(num_components):
        numerator = 0.0
        denominator = 0.0
        for j in range(num_components):
            numerator += pis[j] * math.exp(-compute_KLDiv_Truncated_Gaussians(means_f[i], means_f[j], variances_f[i], variances_f[j]))
            denominator += omegas[j] * math.exp(-compute_KLDiv_Truncated_Gaussians(means_f[i], means_g[j], variances_f[i], variances_g[j]))
        KLdiv_variational += pis[i] * math.log(numerator/denominator)
    return KLdiv_variational 


# Compute improper integration #
def get_improper_integration_value(mu, var, lower_limit, upper_limit):
    if var == 0:
    	var = 1e-10
    def func(x): return math.exp((-(x - mu)**2) / (2 * var))
    val = quad(func, lower_limit, upper_limit)
    return val[0]


# Compute KL divergence between Truncated Gaussian Distributions #
def compute_KLDiv_Truncated_Gaussians(mean1, mean2, variance1, variance2, lower_limit1, upper_limit1, lower_limit2, upper_limit2):
    if variance2 == 0:
    	variance2 = 1e-10
    var_term = (variance1 / (2 * variance2))
    mean_var_term = ((mean1 - mean2)**2 / (2 * variance2))
    log_numerator = get_improper_integration_value(mu=mean2, var=variance2, lower_limit=lower_limit2, upper_limit=upper_limit2)
    log_denominator = get_improper_integration_value(mu=mean1, var=variance1, lower_limit=lower_limit1, upper_limit=upper_limit1)
    if log_numerator == 0:
        log_numerator = 1e-10
    if log_denominator == 0:
        log_denominator = 1e-10
    log_term = math.log(log_numerator / log_denominator) # log(Area_q/Area_p)
    kl_div = (log_term + var_term + mean_var_term - 0.5)
    return kl_div


# Approximation of KL Divergence between mixtures of Truncated Gaussian Distributions # 
# pis: mixing factors corresponding to mixture model f, omegas: mixing factors corresponding to mixing model g #
def compute_KLDiv_mixture_Truncated_Gaussians(num_components, means_f, means_g, variances_f, variances_g, pis, omegas, lower_f, upper_f, lower_g, upper_g):
    KLdiv_variational = 0.0
    for i in range(num_components):
        numerator = 0.0
        denominator = 0.0
        for j in range(num_components):
            numerator += pis[j] * math.exp(-compute_KLDiv_Truncated_Gaussians(means_f[i], means_f[j], variances_f[i], variances_f[j], 
            lower_f[i], upper_f[i], lower_f[j], upper_f[j]))
            denominator += omegas[j] * math.exp(-compute_KLDiv_Truncated_Gaussians(means_f[i], means_g[j], variances_f[i], variances_g[j], 
            lower_f[i], upper_f[i], lower_g[j], upper_g[j]))
        if numerator == 0:
            numerator = 1e-10
        if denominator == 0:
            denominator = 1e-10
        KLdiv_variational += pis[i] * math.log(numerator/denominator)    
    return KLdiv_variational 

    
    
    
    
    
    
    
