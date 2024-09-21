import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import matplotlib.pyplot as plt



# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extract_dir = '/content/drive/My Drive/img_dtlabs'

# Transformações
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Carregar dataset
dataset = datasets.ImageFolder(extract_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Definição da CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, len(dataset.classes))  # Número de classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instanciar modelo
model = SimpleCNN().to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

"""
## BD e VD
"""

def extract_descriptors(model, dataloader):
    model.eval()
    descriptors = []
    labels_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            descriptors.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.concatenate(descriptors), np.concatenate(labels_list)

# Extraindo descritores
descriptors, labels = extract_descriptors(model, dataloader)

# Salvando no banco de dados
with open('descriptors.pkl', 'wb') as f:
    pickle.dump((descriptors, labels), f)


new_image = Image.open('caminho/para/nova/imagem.jpg')
new_image = transform(new_image).unsqueeze(0).to(device)

# Extrair descritor
model.eval()
with torch.no_grad():
    new_descriptor = model(new_image).cpu().numpy()

# Adicionar ao banco de dados
descriptors = np.append(descriptors, new_descriptor, axis=0)

# Atualizar banco de dados
with open('descriptors.pkl', 'wb') as f:
    pickle.dump((descriptors, labels), f)


# Carregar descritores do banco de dados
with open('descriptors.pkl', 'rb') as f:
    descriptors, labels = pickle.load(f)

# Calcular similaridade
similarity_scores = cosine_similarity(new_descriptor, descriptors)
best_match_index = np.argmax(similarity_scores)
best_score = similarity_scores[0][best_match_index]

# Exibir resultados
print(f'Best match index: {best_match_index}, Score: {best_score}')
print(f'Identified as: {labels[best_match_index]}')

# Exibir a imagem
plt.imshow(new_image.cpu().permute(1, 2, 0))
plt.title(f'Match Score: {best_score:.2f}')
plt.axis('off')
plt.show()