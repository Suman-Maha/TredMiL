import torch
import argparse
import os.path as path
from torch.autograd import Variable
from torchvision.utils import save_image
from TrainerFile import TrainerModule
from Utilities import get_config, get_loader, prepare_folder

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_file', type=str, default='demo_data_folder.yaml', help='Path to the Config file')
parser.add_argument('-o', '--output_path', type=str, default='.', help='Output path')
options = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_file = get_config(config=options.config_file)
trainer = TrainerModule(config_file)
trainer = trainer.to(DEVICE)

template_loader = get_loader(config_file, train_data=False, validation_data=False, template_data=True, source_data=False)
print(len(template_loader))
source_loader = get_loader(config_file, train_data=False, validation_data=False, template_data=False, source_data=True)
print(len(source_loader))

model_name = path.splitext(path.basename(options.config_file))[0]
output_dir = path.join(options.output_path + "/outputs", model_name)
checkpoint_dir, image_dir = prepare_folder(output_dir)

print("**** Mapping Operation is ON ****")
sum_color_appearance_code = Variable(torch.zeros(1, trainer.color_appearance_dim, 1, 1))

count = 0
sum_color_appearance_code = 0
for iter_count, (template_image, _) in enumerate(template_loader, 0):
    template_image = template_image.to(DEVICE).detach()
    count = count + 1
    template_color_appearance_code, template_stain_density_code  = trainer.get_template_image_statistic(template_image, checkpoint_dir=checkpoint_dir)
    sum_color_appearance_code += template_color_appearance_code

average_color_apperarace_code = (sum_color_appearance_code / count)

for iter_num, (source_image, sample_name) in enumerate(source_loader, 0):
    source_image = source_image.to(DEVICE).detach()
    file_name = sample_name[0][:-4]
    variant_name = "_Recon.ppm"
    recon_name = file_name + variant_name
    print(recon_name)
    source_color_appearance_code, source_stain_density_code = trainer.get_template_image_statistic(source_image, checkpoint_dir=checkpoint_dir)
    image_recon = trainer.gen.decoder_block(average_color_apperarace_code, source_stain_density_code)
    save_image(image_recon, f"Model_Output/" + recon_name)
    print(f"Operation on image {iter_num + 1} is completed\n")



   
      
