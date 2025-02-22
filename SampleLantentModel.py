import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import guided_diffusion_v3 as gd
from model7 import UNETv13
from encoder_model import Encoder
from decoder_model import Decoder

def create_gaussian_diffusion(steps=1000, noise_schedule="linear"):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    return gd.SpacedDiffusion(
        use_timesteps=gd.space_timesteps(steps, [steps]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_dir = 'D:/DiffusionModels/direccion'
training_epochs = 360

encoder = Encoder().to(device)
decoder = Decoder().to(device)
encoder.load_state_dict(torch.load(f"{save_dir}/encoder.pth", map_location=device))
decoder.load_state_dict(torch.load(f"{save_dir}/decoder.pth", map_location=device))

model = UNETv13(residual=True, attention_res=[], group_norm=True).to(device)
model.load_state_dict(torch.load(f"{save_dir}/model_{training_epochs}.pth", map_location=device))

diffusion = create_gaussian_diffusion()
input_folder = 'D:/DiffusionModels/SmallDataset/rf'
output_folder = 'D:/DiffusionModels/SmallDataset/onepw'
data = gd.CustomDataset(input_folder, output_folder, transform=True)
dataloader = DataLoader(data, batch_size=1, shuffle=True)
x_sample, y_sample = next(iter(dataloader))
x_sample, y_sample = x_sample.to(device), y_sample.to(device)

intermediate = []
for step in diffusion.p_sample_loop_progressive(model, y_sample.shape, x_sample, progress=True, clip_denoised=True):
    intermediate.append(step['sample'].cpu().detach())

intermediate_decoded = [decoder(latent.to(device)).cpu().detach() for latent in intermediate]

def show_tensor_image(image):
    plt.imshow(image.squeeze(0).cpu().numpy(), cmap='gray')
    plt.colorbar()

def show_reverse_process(intermediate_decoded):
    plt.figure(figsize=(15, 2))
    for id, y_gen in enumerate(intermediate_decoded):
        plt.subplot(1, len(intermediate_decoded), id+1)
        show_tensor_image(y_gen)
    plt.show()

show_reverse_process(intermediate_decoded[::100])

plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
show_tensor_image(intermediate_decoded[-1])
plt.title('Diffusion (Reconstruida)')

plt.subplot(1, 2, 2)
show_tensor_image(y_sample.cpu())
plt.title('Original')
plt.show()
