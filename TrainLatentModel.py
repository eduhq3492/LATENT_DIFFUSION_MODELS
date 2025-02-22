import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import guided_diffusion_v3 as gd
from datetime import datetime
import torch.nn.functional as func
from model7 import UNETv13
import torch.nn as nn
from encoder_model import Encoder
from decoder_model import Decoder

TRAIN_PATH = 'D:/DiffusionModels/SmallDataset/rf'
TRAIN_ONEPW_PATH = 'D:/DiffusionModels/SmallDataset/onepw'

save_dir = 'D:/DiffusionModels/direccion'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
decoder = Decoder().to(device)
encoder.load_state_dict(torch.load(f"{save_dir}/encoder.pth", map_location=device), strict=False)
decoder.load_state_dict(torch.load(f"{save_dir}/decoder.pth", map_location=device), strict=False)

class ONEPW_Dataset(Dataset):
    def __init__(self, data, onepw_img):
        self.train_data = data
        self.train_onepw_img = onepw_img
        self.images = sorted(os.listdir(self.train_data))
        self.onepw_images = sorted(os.listdir(self.train_onepw_img))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        rf_image = torch.Tensor(np.load(os.path.join(self.train_data, self.images[idx]))).permute(2, 0, 1)
        onepw_img = torch.Tensor(np.load(os.path.join(self.train_onepw_img, self.onepw_images[idx]))).unsqueeze(0)
        print(f"Shape of onepw_img before permute: {onepw_img.shape}")  # Debugging

        # Asegurar el formato correcto
        onepw_img = onepw_img.permute(0, 2, 1)  # [batch_size, channels, height, width]
        print(f"Shape of onepw_img after permute: {onepw_img.shape}")  # Debugging
        
        with torch.no_grad():
            noise = torch.randn_like(onepw_img)
            latent_representation = encoder(onepw_img.to(device), noise).cpu()
            print(f"Shape of latent_representation: {latent_representation.shape}")  # Debugging
        return rf_image, latent_representation

def main():
    batch_size = 4
    n_epoch = 2
    l_rate = 1e-5
    train_dataset = ONEPW_Dataset(TRAIN_PATH, TRAIN_ONEPW_PATH)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    time_steps = 100
    betas = gd.get_named_beta_schedule('linear', time_steps)
    diffusion = gd.SpacedDiffusion(
        use_timesteps=gd.space_timesteps(time_steps, section_counts=[time_steps]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=True,
    )
    
    nn_model = UNETv13(residual=False, attention_res=[], group_norm=True).to(device)
    optim = torch.optim.Adam(nn_model.parameters(), lr=l_rate)
    loss_arr = []

    for ep in range(1, n_epoch+1):
        for x, y in train_loader:
            print(f"Shape of x: {x.shape}, Shape of y: {y.shape}")  # Debugging
            optim.zero_grad()
            x, y = x.to(device), y.to(device)
            noise = torch.randn_like(y)
            t = torch.randint(0, time_steps, (x.shape[0],)).to(device)
            y_pert = diffusion.q_sample(y, t, noise)
            predicted_noise = nn_model(x, y_pert, t)
            loss = func.mse_loss(predicted_noise, noise)
            loss.backward()
            optim.step()
            loss_arr.append(loss.item())

        print(f'Epoch {ep}/{n_epoch}, Loss: {loss_arr[-1]:.4f}, {datetime.now()}')
        if ep % 5 == 0 or ep == n_epoch:
            torch.save(nn_model.state_dict(), f"{save_dir}/model_{ep}.pth")
            np.save(f"{save_dir}/loss_{ep}.npy", np.array(loss_arr))

if __name__ == '__main__':
    main()
