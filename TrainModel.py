import sys
import torch
import shutil
import argparse
import numpy as np
import tensorboardX
import os.path as path
from itertools import cycle
from TrainerFile import TrainerModule
from pytorchtools import EarlyStopping
from Utilities import get_loader, prepare_folder, write_loss, get_config, write_image_output, TimeElapsed

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_file', type=str, default='ConfigFile.yaml', help='Path to the Config file')
parser.add_argument('-o', '--output_path', type=str, default='.', help='Output path')
parser.add_argument('-r', '--resume', action='store_true')
options = parser.parse_args()

# Load experiment settings #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_file = get_config(config=options.config_file)
maximum_iterations = config_file['max_iter']
display_size = config_file['display_size']
trainer = TrainerModule(config_file)
trainer = trainer.to(DEVICE)

train_loader = get_loader(config_file, train_data=True)
validation_loader = get_loader(config_file, validation_data=True)
print(len(train_loader))
print(len(validation_loader))
train_display_images = torch.stack([train_loader.dataset[i] for i in range(display_size)])
train_display_images = train_display_images.to(DEVICE)

model_name = path.splitext(path.basename(options.config_file))[0]
train_writer = tensorboardX.SummaryWriter(path.join(options.output_path + "/logs", model_name))
output_dir = path.join(options.output_path + "/outputs", model_name)
checkpoint_dir, image_dir = prepare_folder(output_dir)
shutil.copy(options.config_file, path.join(output_dir, 'config.yaml'))

iter_count = trainer.resume_model(checkpoint_dir=checkpoint_dir, hyper_parameters=config_file) if options.resume else 0

train_losses = []
validation_losses = []

average_train_losses = []
average_validation_losses = []

patience = 100
early_stopping = EarlyStopping(patience=patience, verbose=True)

while True:
    for iter_num, (training_images, validation_images) in enumerate(zip(train_loader, cycle(validation_loader))):
        trainer.update_learning_rate()
        training_images = training_images.to(DEVICE).detach()
        validation_images = validation_images.to(DEVICE).detach()

        with TimeElapsed("Elapsed time in Update: %f"):
            #_, _, train_discriminator_loss = trainer.update_discriminator(images)
            #_, _, train_generator_loss = trainer.update_generator(images)
            train_discriminator_loss = trainer.update_discriminator(training_images)
            train_generator_loss = trainer.update_generator(training_images)
            train_loss = (train_discriminator_loss.item() + train_generator_loss.item())
            train_losses.append(train_loss)
            
            #discriminator, optimizer_discriminator, validation_discriminator_loss = trainer.update_discriminator(valid_images, training=False)
            #generator, optimizer_generator, validation_generator_loss = trainer.update_generator(valid_images, training=False)
            validation_discriminator_loss = trainer.update_discriminator(validation_images)
            validation_generator_loss = trainer.update_generator(validation_images)
            generator = trainer.gen
            discriminator = trainer.disc
            optimizer_generator = trainer.opt_gen
            optimizer_discriminator = trainer.opt_disc
            spread_factor = trainer.spread_factor
            validation_loss = (validation_discriminator_loss.item() + validation_generator_loss.item())
            validation_losses.append(validation_loss)
            torch.cuda.synchronize()
            
        train_loss_avg = np.average(train_losses)
        validation_loss_avg = np.average(validation_losses)
        
        average_train_losses.append(train_loss_avg)
        average_validation_losses.append(validation_loss_avg)
        
        train_losses = []
        validation_losses = []
        
        early_stopping(checkpoint_dir, validation_loss, generator, discriminator, optimizer_generator, optimizer_discriminator, spread_factor)
        if early_stopping.early_stop:
            sys.exit('***** Early Stopping Executed *****')

        # Writing in log file #
        if (iter_count + 1) % config_file['log_iter'] == 0:
            print("Iteration counter: %05d/%05d" % ((iter_count + 1), maximum_iterations))
            write_loss(iteration=iter_count, trainer=trainer, train_writer=train_writer)

        # Write images #
        if (iter_count + 1) % config_file['image_save_iter'] == 0:
            with torch.no_grad():
                train_image_outputs = trainer.sample(train_display_images)
            postfix_part1 = 'train%05d' % (iter_count + 1)
            #postfix_part2 = '_k_' + str(spread_factor)
            postfix_part2 = '_k_value'
            postfix_total = postfix_part1 + postfix_part2
            write_image_output(image_output=train_image_outputs, display_image_num=display_size,
                               image_directory=image_dir, postfix=postfix_total)

        if (iter_count + 1) % config_file['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images)
            write_image_output(image_output=image_outputs, display_image_num=display_size, image_directory=image_dir,
                               postfix=postfix_total)

	# Save network parameters #
        if (iter_count + 1) % config_file['snapshot_save_iter'] == 0:
            trainer.save_model_state(checkpoint_dir=checkpoint_dir, spread_factor=spread_factor)

        iter_count += 1
        if iter_count >= maximum_iterations:
            sys.exit('**** Maximum Iterations: Training is FINISHED ****')
            




