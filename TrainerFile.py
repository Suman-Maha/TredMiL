import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optimizer
from torch.autograd import Variable
from Networks import Generator, Discriminator
from Sample_from_tGMM import extract_sample_mixture_truncated_gaussian, extract_mean_variance
from Utilities import get_scheduler, get_model_list, weights_init, compute_KLDiv_mixture_Truncated_Gaussians, compute_KLDiv_multivariate_Gaussians_standard_normal


class TrainerModule(nn.Module):
    def __init__(self, hyper_parameters):
        super(TrainerModule, self).__init__()
        self.type_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.color_appearance_dim = hyper_parameters['gen']['color_appearance_dim']
        self.gan_loss_weight = hyper_parameters['gan_w']
        self.recon_loss_weight = hyper_parameters['recon_w']
        self.recon_color_appearance_weight = hyper_parameters['recon_c_a_w']
        self.recon_stain_density_weight = hyper_parameters['recon_s_d_w']
        self.regularization_weight = hyper_parameters['regularization_w']
        input_dim = hyper_parameters['input_dim']
        gen_dim = hyper_parameters['gen']['dim']
        num_down_block = hyper_parameters['gen']['num_down_block']
        num_down = hyper_parameters['gen']['num_down']
        num_res = hyper_parameters['gen']['num_res']
        mlp_dim = hyper_parameters['gen']['mlp_dim']
        disc_dim = hyper_parameters['disc']['dim']
        num_layers = hyper_parameters['disc']['num_layers']
        num_scales = hyper_parameters['disc']['num_scales']
        learning_rate = hyper_parameters['lr']
        beta1 = hyper_parameters['beta1']
        beta2 = hyper_parameters['beta2']
        weight_decay = hyper_parameters['weight_decay']
        init_network = hyper_parameters['init']
        display_size = hyper_parameters['display_size']
        self.num_components = hyper_parameters['gen']['num_components']
        self.mean_H = hyper_parameters['H_stain_mean']
        self.mean_E = hyper_parameters['E_stain_mean']
        self.scale_param = hyper_parameters['scale_std']
        self.spread_factor = hyper_parameters['spread_factor']

        # Initialize the Generator and Discriminator module #
        self.gen = Generator(input_dim=input_dim, dim=gen_dim, color_dim=self.color_appearance_dim, num_down_block=num_down_block, 
        num_down=num_down, num_res=num_res, mlp_dim=mlp_dim)
        self.gen = self.gen.to(self.type_device)
        self.disc = Discriminator(input_dim=input_dim, dim=disc_dim, num_layers=num_layers, num_scales=num_scales, use_leaky=True)
        self.disc = self.disc.to(self.type_device)

        # Set up the optimizers and schedulers #
        gen_params = list(self.gen.parameters())
        disc_params = list(self.disc.parameters())
        self.opt_gen = optimizer.Adam([param for param in gen_params if param.requires_grad], lr=learning_rate,
                                      betas=(beta1, beta2), weight_decay=weight_decay)
        self.opt_disc = optimizer.Adam([param for param in disc_params if param.requires_grad], lr=learning_rate,
                                       betas=(beta1, beta2), weight_decay=weight_decay)
        self.gen_scheduler = get_scheduler(optimizer=self.opt_gen, hyper_parameters=hyper_parameters)
        self.disc_scheduler = get_scheduler(optimizer=self.opt_disc, hyper_parameters=hyper_parameters)

        # Initialize the network weights #
        self.apply(weights_init(init_type=init_network))
        self.gen.apply(weights_init(init_type='gaussian'))
        self.disc.apply(weights_init(init_type='gaussian'))

        # Create random color code #
        self.means_prior = list()
        self.means_prior.append(self.mean_H)
        self.means_prior.append(self.mean_E)
        self.left_extreme = (self.means_prior[0] - (self.spread_factor * self.scale_param))
        self.right_extreme = (self.means_prior[1] + (self.spread_factor * self.scale_param))
        #self.left_extreme = -math.inf
        #self.right_extreme = math.inf
        self.color_appearance_random = torch.tensor(extract_sample_mixture_truncated_gaussian(self.means_prior, self.scale_param, 
        self.color_appearance_dim, self.left_extreme, self.right_extreme, display_size))

    def reconstruction_criterion(self, source, target):
        return torch.mean((target - source)**2)

    def forward(self, x):
        self.eval()
        #x = x.to(self.type_device)
        color_appearance_random = Variable(self.color_appearance_random)
        color_appearance_random = color_appearance_random.to(self.type_device)
        color_appearance_code, stain_density_code = self.gen.encoder_block(x)
        x_recon_color_appearance_random = self.gen.decoder_block(color_appearance_random, stain_density_code)
        self.train()
        return x_recon_color_appearance_random

    def sample(self, x):
        self.eval()
        #x = x.to(self.type_device)
        color_appearance_rand = Variable(self.color_appearance_random)
        color_appearance_rand = color_appearance_rand.to(self.type_device)
        color_appearance_random = Variable(torch.tensor(extract_sample_mixture_truncated_gaussian(self.means_prior, self.scale_param, 
        self.color_appearance_dim, self.left_extreme, self.right_extreme, x.size(0))))
        color_appearance_random = color_appearance_random.to(self.type_device)
        x_recon = []
        x_rand_color_appearance_recon = []
        x_random_color_appearance_recon = []
        for i in range(x.size(0)):
            color_appearance_code, stain_density_code = self.gen.encoder_block(x[i].unsqueeze(0))
            x_recon.append(self.gen.decoder_block(color_appearance_code, stain_density_code))
            x_rand_color_appearance_recon.append(self.gen.decoder_block(color_appearance_rand[i].float().unsqueeze(0), stain_density_code))
            x_random_color_appearance_recon.append(self.gen.decoder_block(color_appearance_random[i].float().unsqueeze(0), stain_density_code))
        x_recon = torch.cat(x_recon)
        x_rand_color_appearance_recon = torch.cat(x_rand_color_appearance_recon)
        x_random_color_appearance_recon = torch.cat(x_random_color_appearance_recon)
        self.train()
        return x_recon, x_rand_color_appearance_recon, x_random_color_appearance_recon

    def update_discriminator(self, x):
        #if training:
        #    self.opt_disc.zero_grad() 
        self.opt_disc.zero_grad()
        #x = x.to(self.type_device)
        color_appearance_rand = Variable(torch.tensor(extract_sample_mixture_truncated_gaussian(self.means_prior, self.scale_param, 
        self.color_appearance_dim, self.left_extreme, self.right_extreme, x.size(0))))
        color_appearance_rand = color_appearance_rand.to(self.type_device)
        # Encode #
        color_appearance_code, stain_density_code = self.gen.encoder_block(x.detach())
        # Decode with random color appearance code #
        x_recon_rand_color_appearance = self.gen.decoder_block(color_appearance_rand.float(), stain_density_code)
        loss_gan_disc = self.disc.calculate_discriminator_loss(x, x_recon_rand_color_appearance.detach())
        # Compute total discriminator loss #
        loss_discriminator_total = (self.gan_loss_weight * loss_gan_disc)
        # Backpropagation: backward calculation & weight update #
        #if training:
        #    loss_discriminator_total.backward()
        #    self.opt_disc.step()
        #return self.disc, self.opt_disc, loss_discriminator_total
        loss_discriminator_total.backward()
        self.opt_disc.step()
        return loss_discriminator_total

    def update_generator(self, x):
        #if training:
        #    self.opt_gen.zero_grad()
        self.opt_gen.zero_grad()
        #x = x.to(self.type_device)
        color_appearance_rand = Variable(torch.tensor(extract_sample_mixture_truncated_gaussian(self.means_prior, self.scale_param, 
        self.color_appearance_dim, self.left_extreme, self.right_extreme, x.size(0))))
        color_appearance_rand = color_appearance_rand.to(self.type_device)
        # Encode #
        color_appearance_code, stain_density_code = self.gen.encoder_block(x)
        color_appearance_code_reshaped = color_appearance_code.reshape([-1])
        mean_stain1, mean_stain2, var_stain1, var_stain2, mixing_stain1, mixing_stain2 = extract_mean_variance(color_appearance_code_reshaped)
        means_estimated = list()
        means_estimated.append(mean_stain1)
        means_estimated.append(mean_stain2)
        var_stain_estimated = list()
        var_stain_estimated.append(var_stain1)
        var_stain_estimated.append(var_stain2)
        mixing_factor_estimated = list()
        mixing_factor_estimated.append(mixing_stain1)
        mixing_factor_estimated.append(mixing_stain2)
        var_stain_prior = list()
        var_stain_prior.append(self.scale_param)
        var_stain_prior.append(self.scale_param)
        mixing_factor_prior = list()
        mixing_prior_stain1 = np.random.uniform(0.45, 0.55)
        mixing_prior_stain2 = (1 - mixing_prior_stain1)
        mixing_factor_prior.append(mixing_prior_stain1)
        mixing_factor_prior.append(mixing_prior_stain2)
        lower_prior = list()
        lower_prior.append(self.means_prior[0] - (self.spread_factor * var_stain_prior[0]))
        #lower_prior.append(-math.inf)
        lower_prior.append(-math.inf)
        upper_prior = list()
        upper_prior.append(math.inf)
        upper_prior.append(self.means_prior[1] + (self.spread_factor * var_stain_prior[1]))
        #upper_prior.append(math.inf)
        lower_estimated = list()
        lower_estimated.append(means_estimated[0] - (self.spread_factor * var_stain_estimated[0]))
        #lower_estimated.append(-math.inf)
        lower_estimated.append(-math.inf)
        upper_estimated = list()
        upper_estimated.append(math.inf)
        upper_estimated.append(means_estimated[1] + (self.spread_factor * var_stain_estimated[1]))
        #upper_estimated.append(math.inf)
        KL_Div_color_appearance = compute_KLDiv_mixture_Truncated_Gaussians(self.num_components, self.means_prior, means_estimated, var_stain_prior, 
        var_stain_estimated, mixing_factor_prior, mixing_factor_estimated, lower_prior, upper_prior, lower_estimated, upper_estimated)
        #print("KL Divergence Color Appearance: ", KL_Div_color_appearance)
        if math.isinf(KL_Div_color_appearance):
            self.gen.apply(weights_init(init_type='gaussian'))
            self.disc.apply(weights_init(init_type='gaussian'))
        #stain_density_code_reshaped = stain_density_code.reshape((256, 4096))
        #stain_density_code_reshaped = stain_density_code_reshaped.detach().numpy()
        #stain_density_code_mean = stain_density_code_reshaped.mean(axis=1)
        #stain_density_code_covariance = np.cov(stain_density_code_reshaped)
        #KL_Div_stain_density = compute_KLDiv_multivariate_Gaussians_standard_normal(stain_density_code_mean, stain_density_code_covariance)
        #print("KL Divergence Stain Density: ", KL_Div_stain_density)

        stain_density_rand = Variable(torch.randn(1, 256, 64, 64))
        stain_density_rand = stain_density_rand.to(self.type_device)
        log_stain_density_code = torch.log(stain_density_code)
        regulization_term = torch.matmul(stain_density_rand, log_stain_density_code)
        square_regularization_term = torch.square(regulization_term)
        sum_square_regularization_term = torch.sum(square_regularization_term)
        # Decode #
        x_recon = self.gen.decoder_block(color_appearance_code, stain_density_code)
        # Decode (with random color appearance code) #
        x_recon_color_appearance_rand = self.gen.decoder_block(color_appearance_rand.float(), stain_density_code)
        # Encode again to get cycle loss #
        color_appearance_code_hat, stain_density_code_hat = self.gen.encoder_block(x_recon_color_appearance_rand)
        # Compute reconstruction losses #
        loss_recon_x = self.reconstruction_criterion(x, x_recon)
        loss_recon_color_appearance = self.reconstruction_criterion(color_appearance_code, color_appearance_code_hat)
        loss_recon_stain_density = self.reconstruction_criterion(stain_density_code, stain_density_code_hat)
        # Compute regularization loss #
        color_appearance_code_reshaped = color_appearance_code_reshaped.cpu().detach().numpy()
        #regularization_loss_color_appearance = (color_appearance_code_reshaped @ np.log(color_appearance_code_reshaped))
        #stain_density_code_1D_reshaped = stain_density_code_reshaped.reshape([-1])
        #regularization_loss_stain_density = (stain_density_code_1D_reshaped @ np.log(stain_density_code_1D_reshaped))
        # Compute GAN loss #
        loss_gan_gen = self.disc.calculate_generator_loss(x_recon_color_appearance_rand)
        # Compute total loss #
        weighted_loss_x = (self.recon_loss_weight * loss_recon_x)
        weighted_recon_color_appearance = (self.recon_color_appearance_weight * loss_recon_color_appearance) 
        weighted_recon_stain_density = (self.recon_stain_density_weight * loss_recon_stain_density)
        #weighted_loss_gan = (self.gan_loss_weight * loss_gan_gen)
        #weighted_divergence_loss = (self.regularization_weight * KL_Div_color_appearance)
        reconstruction_factors = (weighted_loss_x + weighted_recon_color_appearance + weighted_recon_stain_density)
        #reconstruction_factors = weighted_loss_x
        #divergence_measures = (KL_Div_color_appearance + KL_Div_stain_density)
        divergence_measures = (self.regularization_weight * KL_Div_color_appearance)
        #entropy_regularizers = (regularization_loss_color_appearance + regularization_loss_stain_density)
        #entropy_regularizers = 0.0
        #regularization_factor = - sum_square_regularization_term

        generalization_loss = (self.gan_loss_weight * loss_gan_gen)
        #reconstruction_loss = (self.recon_loss_weight * (reconstruction_factors + divergence_measures + regularization_factor))
        #reconstruction_loss = (self.recon_loss_weight * (reconstruction_factors + divergence_measures))
        reconstruction_loss = (reconstruction_factors + divergence_measures)
        #reconstruction_loss = reconstruction_factors
        #reconstruction_loss = (self.recon_loss_weight * ((reconstruction_factors + divergence_measures) - entropy_regularizers))
        loss_generator_total = (generalization_loss + reconstruction_loss)
        #loss_generator_total = generalization_loss
        #if training:
        #    loss_generator_total.backward()
        #    self.opt_gen.step()
        #return self.gen, self.opt_gen, loss_generator_total
        loss_generator_total.backward()
        self.opt_gen.step()
        return loss_generator_total

    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.disc_scheduler is not None:
            self.disc_scheduler.step()

    def resume_model_state(self, checkpoint_dir, hyper_parameters):
        # Load Generators #
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict_gen = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict_gen['gen_state_dict'])
        iteration = int(last_model_name[-11:-3])
        # Load Discriminators #
        last_model_model = get_model_list(checkpoint_dir, "dis")
        state_dict_disc = torch.load(last_model_model)
        self.disc.load_state_dict(state_dict_disc['dis_state_dict'])
        # Load Optimizers #
        state_dict_opt = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        self.opt_gen.load_state_dict(state_dict_opt['gen'])
        self.opt_disc.load_state_dict(state_dict_opt['dis'])
        # Reinitialize schedulers #
        self.gen_scheduler = get_scheduler(optimizer=self.opt_gen, hyper_parameters=hyper_parameters,
                                           iterations=iteration)
        self.disc_scheduler = get_scheduler(optimizer=self.opt_disc, hyper_parameters=hyper_parameters,
                                            iterations=iteration)
        print('Resume from iteration %d' % iteration)
        return iteration

    def save_model_state(self, checkpoint_dir, spread_factor):
        # Save Generators, Discriminators and Optimizers #
        #gen_name = os.path.join(checkpoint_dir, 'gen_100th_nsgan_k_' + str(spread_factor) + '.pt')
        #disc_name = os.path.join(checkpoint_dir, 'disc_100th_nsgan_k_' + str(spread_factor) + '.pt')
        #opt_name = os.path.join(checkpoint_dir, 'optimizer_100th_nsgan_k_' + str(spread_factor) + '.pt')
        gen_name = os.path.join(checkpoint_dir, 'gen_100th_k_value.pt')
        disc_name = os.path.join(checkpoint_dir, 'disc_100th_k_value.pt')
        opt_name = os.path.join(checkpoint_dir, 'optimizer_100th_k_value.pt')
        print('Multiple of 100 iterations encountered. Saving 100th Iteration Model State ...')
        torch.save({'gen_state_dict': self.gen.state_dict()}, gen_name)
        torch.save({'dis_state_dict': self.disc.state_dict()}, disc_name)
        torch.save({'gen': self.opt_gen.state_dict(), 'dis': self.opt_disc.state_dict()}, opt_name)

    def get_template_image_statistic(self, x, checkpoint_dir):
        trained_gen = get_model_list(checkpoint_dir, "gen_100th_k_value")
        state_dict_trained_gen = torch.load(trained_gen)
        self.gen.load_state_dict(state_dict_trained_gen['gen_state_dict'])
        template_color_appearance_code, template_stain_density_code = self.gen.encoder_block(x)
        return template_color_appearance_code, template_stain_density_code  







