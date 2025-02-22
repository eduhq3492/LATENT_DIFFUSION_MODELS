import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import guided_diffusion_v3 as gd
from encoder_model import Encoder
from decoder_model import Decoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)  
decoder = Decoder().to(device)  

transform = transforms.Compose([
    transforms.Resize((800, 128)),  
    transforms.ToTensor(), 
])
input_folder = 'D:/DiffusionModels/SmallDataset/rf'
output_folder = 'D:/DiffusionModels/SmallDataset/onepw'
data = gd.CustomDataset(input_folder, output_folder, transform=transform)
BATCH_SIZE = 8
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
criterion = nn.MSELoss()

def show_tensor_image(tensor_img, title=""):
    img = tensor_img.squeeze(0).cpu().detach().numpy()
    plt.imshow(img, cmap='gray', aspect='auto')
    plt.colorbar()
    plt.axis("off")
    plt.title(title)

EPOCHS = 2
for epoch in range(EPOCHS):
    total_loss = 0
    for _, y_sample in dataloader:
        y_sample = y_sample.to(device)
        print(f"Input shape antes del encoder: {y_sample.shape}") 

        with torch.no_grad():
            noise = torch.randn_like(y_sample)  
            for _ in range(3):  
                noise = nn.functional.avg_pool2d(noise, kernel_size=2, stride=2)
        
        latent = encoder(y_sample, noise)
        print(f"Latent shape antes de Decoder: {latent.shape}")  
        
        if latent.shape[1] != 4:
            print(f"Ajustando canales de latent: {latent.shape[1]} → 4")
            latent = nn.Conv2d(latent.shape[1], 4, kernel_size=1, padding=0).to(device)(latent)
        
        reconstructed = decoder(latent)
        print(f"Output shape después de Decoder: {reconstructed.shape}") 
        if reconstructed.shape != y_sample.shape:
            print(f"Ajustando tamaño de salida: {reconstructed.shape} → {y_sample.shape}")
            reconstructed = nn.functional.interpolate(reconstructed, size=y_sample.shape[2:], mode="bilinear", align_corners=False)

        # Cálculo de la pérdida
        loss = criterion(reconstructed, y_sample)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Época {epoch}/{EPOCHS}, Pérdida: {total_loss / len(dataloader):.4f}")

    # Ploteos
    plt.figure(figsize=(9, 3))

    plt.subplot(1, 2, 1)
    show_tensor_image(y_sample[0], "Original")

    plt.subplot(1, 2, 2)
    show_tensor_image(reconstructed[0], "Reconstruida")

    plt.show()

save_dir = 'D:/DiffusionModels/direccion'
torch.save(encoder.state_dict(), f"{save_dir}/encoder.pth")
torch.save(decoder.state_dict(), f"{save_dir}/decoder.pth")

print("Entrenamiento completado y modelos guardados.")






