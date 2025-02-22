import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import guided_diffusion_v3 as gd  # Tu librería de difusión personalizada

# 🔹 Importar Encoder y Decoder
from autoencoder import Encoder, Decoder  

# 🔹 Cargar el modelo de difusión
def create_gaussian_diffusion(steps=1000, noise_schedule="linear", rescale_timesteps=False):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    return gd.SpacedDiffusion(
        use_timesteps=gd.space_timesteps(steps, [steps]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=rescale_timesteps,
    )

diffusion = create_gaussian_diffusion()

# 🔹 Configuración del dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 🔹 Cargar Dataset
input_folder = 'D:/DiffusionModels/SmallDataset/rf'
output_folder = 'D:/DiffusionModels/SmallDataset/onepw'
data = gd.CustomDataset(input_folder, output_folder, transform=True)
BATCH_SIZE = 1
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# 🔹 Cargar Modelo UNet entrenado
from model7 import UNETv13
save_dir = 'D:/DiffusionModels/direccion'
training_epochs = 360
model13A = UNETv13(residual=True, attention_res=[], group_norm=True).to(device)
checkpoint = torch.load(f"{save_dir}/model_{training_epochs}.pth", map_location=device)
model13A.load_state_dict(checkpoint)
model13A.eval()

# 🔹 Cargar Encoder y Decoder entrenados
latent_dim = 128
encoder = Encoder(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)
encoder.load_state_dict(torch.load(f"{save_dir}/encoder.pth", map_location=device))
decoder.load_state_dict(torch.load(f"{save_dir}/decoder.pth", map_location=device))
encoder.eval()
decoder.eval()

# 🔹 Obtener un batch de datos
x_sample, y_sample = next(iter(dataloader))
x_sample = x_sample.to(device)
y_sample = y_sample.to(device)

# 🔹 Codificar en el espacio latente
z_sample = encoder(y_sample)

# 🔹 Aplicar proceso de difusión en el espacio latente
intermediate = []
for step in diffusion.p_sample_loop_progressive(model13A, z_sample.shape, z_sample, progress=True, clip_denoised=True):
    intermediate.append(step['sample'].cpu().detach())

# 🔹 Decodificar el resultado del espacio latente al espacio de píxeles
generated_image = decoder(intermediate[-1].to(device))

# 🔹 Funciones para visualización
def show_tensor_image(image, title="Generated Image"):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW → HWC
        transforms.Lambda(lambda t: (t * 60) - 60.),
        transforms.Lambda(lambda t: t.numpy())
    ])
    plt.imshow(reverse_transforms(image), cmap='gray', extent=[-20,20,50,0])
    plt.colorbar()
    plt.clim(-60, 0)
    plt.title(title)

# 🔹 Mostrar el proceso de reversión
def show_reverse_process(intermediate):
    num_intermediate = len(intermediate)
    plt.figure(figsize=(15, 2))
    plt.axis('off')
    for id, z_gen in enumerate(intermediate):
        plt.subplot(1, num_intermediate, id+1)
        show_tensor_image(z_gen, title=f"Step {id}")
    plt.show()

# 🔹 Visualizar resultados
plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
show_tensor_image(generated_image.cpu(), title="Generated Image")

plt.subplot(1, 2, 2)
show_tensor_image(y_sample.cpu(), title="Original Image")

plt.show()


"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import guided_diffusion_v3 as gd
from model7 import UNETv13
from autoencoder import Encoder, Decoder  # Importamos el autoencoder

# Cargar dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Cargar dataset
input_folder = 'D:/DiffusionModels/SmallDataset/rf'
output_folder = 'D:/DiffusionModels/SmallDataset/onepw'
data = gd.CustomDataset(input_folder, output_folder, transform=True)

BATCH_SIZE = 1
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
x_sample, y_sample = next(iter(dataloader))
x_sample, y_sample = x_sample.to(device), y_sample.to(device)

# Cargar modelo de difusión
def create_gaussian_diffusion(steps=1000, noise_schedule="linear"):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    return gd.SpacedDiffusion(
        use_timesteps=gd.space_timesteps(steps, [steps]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False
    )

diffusion = create_gaussian_diffusion()

# Cargar modelo UNET entrenado
save_dir = 'D:/DiffusionModels/direccion'
training_epochs = 360
model13A = UNETv13(residual=True, attention_res=[], group_norm=True).to(device)
checkpoint = torch.load(f"{save_dir}/model_{training_epochs}.pth", map_location=device)
model13A.load_state_dict(checkpoint)

# Cargar autoencoder
latent_dim = 128
encoder = Encoder(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)
encoder.load_state_dict(torch.load(f"{save_dir}/encoder.pth", map_location=device))
decoder.load_state_dict(torch.load(f"{save_dir}/decoder.pth", map_location=device))

# **1. Codificar la imagen en espacio latente**
latent_y = encoder(y_sample)

# **2. Aplicar el modelo de difusión en el espacio latente**
intermediate = []
for step in diffusion.p_sample_loop_progressive(model13A, latent_y.shape, x_sample, progress=True, clip_denoised=True):
    intermediate.append(step['sample'].cpu().detach())

# **3. Decodificar la imagen generada**
generated_y = decoder(intermediate[-1].to(device))

# **4. Visualizar resultados**
plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
plt.imshow(generated_y.cpu().detach().numpy()[0, 0], cmap='gray')
plt.title('Generado')

plt.subplot(1, 2, 2)
plt.imshow(y_sample.cpu().numpy()[0, 0], cmap='gray')
plt.title('Real')

plt.show()
"""