import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import guided_diffusion_v3 as gd
from model import Encoder, Decoder  # Importamos las clases desde model.py

# 🔹 Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Definir parámetros del autoencoder
latent_dim = 128
input_shape = (1, 800, 128)  

# 🔹 Instanciar Encoder y Decoder usando `model.py`
encoder = Encoder(
    ch=64, out_ch=latent_dim, in_channels=1, resolution=800,
    ch_mult=(1, 2, 4), num_res_blocks=2, attn_resolutions=[],
    z_channels=latent_dim, double_z=False
).to(device)

decoder = Decoder(
    ch=64, out_ch=1, in_channels=latent_dim, resolution=800,
    ch_mult=(1, 2, 4), num_res_blocks=2, attn_resolutions=[],
    z_channels=latent_dim
).to(device)

# 🔹 Transformación de imágenes
transform = transforms.Compose([
    transforms.Resize((800, 128)),  
    transforms.ToTensor()
])

# 🔹 Cargar dataset con la transformación
input_folder = 'D:/DiffusionModels/SmallDataset/rf'
output_folder = 'D:/DiffusionModels/SmallDataset/onepw'
data = gd.CustomDataset(input_folder, output_folder, transform=transform)

BATCH_SIZE = 8
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# 🔹 Configurar optimizador y pérdida
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
criterion = nn.MSELoss()

# 🔹 Función para mostrar imágenes
def show_tensor_image(tensor_img, title=""):
    img = tensor_img.squeeze(0).squeeze(0).cpu().detach().numpy()
    plt.imshow(img, cmap='gray', aspect='auto')
    plt.colorbar()
    plt.axis("off")
    plt.title(title)

# 🔹 Entrenamiento
EPOCHS = 10
for epoch in range(EPOCHS):
    total_loss = 0
    for _, y_sample in dataloader:
        y_sample = y_sample.to(device)
        
        # 🔹 Asegurar que y_sample tiene solo 1 canal
        if y_sample.shape[1] > 1:
            y_sample = y_sample[:, 0:1, :, :]  # Tomar solo el primer canal
        
        optimizer.zero_grad()
        
        # Codificación y decodificación con el nuevo modelo
        latent = encoder(y_sample)
        reconstructed = decoder(latent)
        
        # Cálculo de la pérdida
        loss = criterion(reconstructed, y_sample)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Época {epoch+1}/{EPOCHS}, Pérdida: {total_loss / len(dataloader):.4f}")

    # 🔹 Ploteo de una imagen real vs reconstruida
    plt.figure(figsize=(9, 3))

    plt.subplot(1, 2, 1)
    show_tensor_image(y_sample[0], "Original")

    plt.subplot(1, 2, 2)
    show_tensor_image(reconstructed[0], "Reconstruida")

    plt.show()

# 🔹 Guardar modelos entrenados
save_dir = 'D:/DiffusionModels/direccion'
torch.save(encoder.state_dict(), f"{save_dir}/encoder.pth")
torch.save(decoder.state_dict(), f"{save_dir}/decoder.pth")

print("✅ Entrenamiento completado y modelos guardados.")
