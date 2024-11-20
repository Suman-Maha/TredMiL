import os
import math
import torch
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'
            trace_func (function): trace print function. Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.validation_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, checkpoint_dir, validation_loss, generator, discriminator, optimizer_generator, optimizer_discriminator, spread_factor):

        score = -validation_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.save_model(checkpoint_dir, validation_loss, generator, discriminator, optimizer_generator, optimizer_discriminator, spread_factor)
        elif score < self.best_score + self.delta or math.isnan(score):
            self.counter += 1
            self.trace_func(f'Early Stopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.save_model(checkpoint_dir, validation_loss, generator, discriminator, optimizer_generator, optimizer_discriminator, spread_factor)
            self.counter = 0

    def save_model(self, checkpoint_dir, validation_loss, generator, discriminator, optimizer_generator, optimizer_discriminator, spread_factor):
        # Save Generators, Discriminators and Optimizers #
        #gen_name = os.path.join(checkpoint_dir, 'gen_optimal_nsgan_k_' + str(spread_factor) + '.pt')
        #disc_name = os.path.join(checkpoint_dir, 'disc_optimal_nsgan_k_' + str(spread_factor) + '.pt')
        #opt_name = os.path.join(checkpoint_dir, 'optimizer_optimal_nsgan_k_' + str(spread_factor) + '.pt')
        gen_name = os.path.join(checkpoint_dir, 'gen_optimal_k_value.pt')
        disc_name = os.path.join(checkpoint_dir, 'disc_optimal_k_value.pt')
        opt_name = os.path.join(checkpoint_dir, 'optimizer_optimal_k_value.pt')
        # Saves model when validation loss decrease #
        if self.verbose:
            self.trace_func(f'Validation Loss Decreased ({self.validation_loss_min:.4f} --> {validation_loss:.4f}).  Saving Model States ...')
        torch.save({'gen_state_dict': generator.state_dict()}, gen_name)
        torch.save({'dis_state_dict': discriminator.state_dict()}, disc_name)
        torch.save({'gen': optimizer_generator.state_dict(), 'dis': optimizer_discriminator.state_dict()}, opt_name)
        self.validation_loss_min = validation_loss
        
        
        
        
        
        
        
        
        
               
