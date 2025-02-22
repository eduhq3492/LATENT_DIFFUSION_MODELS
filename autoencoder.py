import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import guided_diffusion_v3 as gd

# 🔹 Definir el Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim=128, input_shape=(1, 800, 128)):  
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))  # 🔹 Forzamos salida (128, 8, 8)
        )

        # **Calculamos automáticamente la entrada de la capa `fc`**
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  
            conv_out = self.conv_layers(dummy_input)
            self.flatten_size = conv_out.view(1, -1).shape[1]  # 🔹 Ahora será 128 * 8 * 8 = 8192

        self.fc = nn.Linear(self.flatten_size, latent_dim)  

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)  # Aplanar
        x = self.fc(x)
        return x

# 🔹 Definir el Decoder para salida (1, 800, 128)
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_shape=(1, 800, 128)):  
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)  # Ajuste para coincidir con la salida del Encoder

        self.model = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 8, 8)  
        x = self.model(x)
        x = torch.nn.functional.interpolate(x, size=(800, 128), mode="bilinear", align_corners=False)  # 🔹 Ajuste final
        return x

# 🔹 Configurar entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
encoder = Encoder(latent_dim, input_shape=(1, 800, 128)).to(device)  
decoder = Decoder(latent_dim, output_shape=(1, 800, 128)).to(device)

# 🔹 Transformación para redimensionar imágenes a (800, 128)
transform = transforms.Compose([
    transforms.Resize((800, 128)),  # 🔹 Redimensionamos a (800, 128)
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

# 🔹 Entrenamiento
EPOCHS = 10
for epoch in range(EPOCHS):
    total_loss = 0
    for _, y_sample in dataloader:
        y_sample = y_sample.to(device)
        optimizer.zero_grad()
        
        # Codificación y decodificación
        latent = encoder(y_sample)
        reconstructed = decoder(latent)
        
        # Cálculo de la pérdida
        loss = criterion(reconstructed, y_sample)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Época {epoch+1}/{EPOCHS}, Pérdida: {total_loss / len(dataloader):.4f}")

# 🔹 Guardar modelos entrenados
save_dir = 'D:/DiffusionModels/direccion'
torch.save(encoder.state_dict(), f"{save_dir}/encoder.pth")
torch.save(decoder.state_dict(), f"{save_dir}/decoder.pth")

print(" Entrenamiento completado y modelos guardados.")










