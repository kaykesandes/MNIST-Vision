# 🔧 Personalização e Modificações

## 📚 Índice
- [Modificações Básicas](#modificações-básicas)
- [Arquitetura](#arquitetura)
- [Hyperparâmetros](#hyperparâmetros)
- [Visualização](#visualização)
- [Datasets](#datasets)
- [Experimentos Avançados](#experimentos-avançados)

## 🎯 Modificações Básicas

### ⚙️ Configurações Rápidas

#### 1. **Alterar Número de Epochs:**
```python
# Em treino_com_visualizacao()
EPOCHS = 10  # ao invés de 3
```

#### 2. **Mudar Batch Size:**
```python
# Na criação dos DataLoaders
trainloader = torch.utils.data.DataLoader(trainset, 
                                         batch_size=128,  # ao invés de 64
                                         shuffle=True)
```

#### 3. **Ajustar Learning Rate:**
```python
# No otimizador
otimizador = optim.SGD(modelo.parameters(), 
                      lr=0.1,      # ao invés de 0.01
                      momentum=0.5)
```

#### 4. **Frequência de Visualização:**
```python
# Em treino_com_visualizacao()
if batches_processados % 100 == 0:  # ao invés de 300
    # Visualização mais frequente
```

## 🏗️ Arquitetura

### 🧠 Modificar Tamanhos das Camadas

#### Rede Mais Simples:
```python
class ModeloSimples(nn.Module):
    def __init__(self):
        super(ModeloSimples, self).__init__()
        self.linear1 = nn.Linear(28*28, 64)   # Menor
        self.linear2 = nn.Linear(64, 10)      # Direta para saída
        
    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = self.linear2(X)
        return F.log_softmax(X, dim=1)
```

#### Rede Mais Complexa:
```python
class ModeloComplexo(nn.Module):
    def __init__(self):
        super(ModeloComplexo, self).__init__()
        self.linear1 = nn.Linear(28*28, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 10)
        
    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = F.relu(self.linear3(X))
        X = F.relu(self.linear4(X))
        X = self.linear5(X)
        return F.log_softmax(X, dim=1)
```

### 🎨 Diferentes Funções de Ativação

#### Sigmoid:
```python
def forward(self, X):
    X = torch.sigmoid(self.linear1(X))
    X = torch.sigmoid(self.linear2(X))
    X = self.linear3(X)
    return F.log_softmax(X, dim=1)
```

#### Tanh:
```python
def forward(self, X):
    X = torch.tanh(self.linear1(X))
    X = torch.tanh(self.linear2(X))
    X = self.linear3(X)
    return F.log_softmax(X, dim=1)
```

#### Leaky ReLU:
```python
def forward(self, X):
    X = F.leaky_relu(self.linear1(X), negative_slope=0.01)
    X = F.leaky_relu(self.linear2(X), negative_slope=0.01)
    X = self.linear3(X)
    return F.log_softmax(X, dim=1)
```

### 🛡️ Adicionar Dropout

```python
class ModeloComDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ModeloComDropout, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = self.dropout(X)  # Dropout após primeira camada
        X = F.relu(self.linear2(X))
        X = self.dropout(X)  # Dropout após segunda camada
        X = self.linear3(X)
        return F.log_softmax(X, dim=1)
```

### 📊 Batch Normalization

```python
class ModeloComBatchNorm(nn.Module):
    def __init__(self):
        super(ModeloComBatchNorm, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, 10)
        
    def forward(self, X):
        X = F.relu(self.bn1(self.linear1(X)))
        X = F.relu(self.bn2(self.linear2(X)))
        X = self.linear3(X)
        return F.log_softmax(X, dim=1)
```

## ⚙️ Hyperparâmetros

### 🎯 Diferentes Otimizadores

#### Adam:
```python
otimizador = optim.Adam(modelo.parameters(), 
                       lr=0.001,
                       betas=(0.9, 0.999),
                       eps=1e-08)
```

#### RMSprop:
```python
otimizador = optim.RMSprop(modelo.parameters(),
                          lr=0.01,
                          alpha=0.99,
                          eps=1e-08)
```

#### AdaGrad:
```python
otimizador = optim.Adagrad(modelo.parameters(),
                          lr=0.01,
                          lr_decay=0,
                          eps=1e-10)
```

### 📈 Learning Rate Scheduling

#### StepLR:
```python
from torch.optim.lr_scheduler import StepLR

otimizador = optim.SGD(modelo.parameters(), lr=0.1)
scheduler = StepLR(otimizador, step_size=30, gamma=0.1)

# No loop de treinamento:
for epoch in range(EPOCHS):
    # ... treinamento ...
    scheduler.step()  # Reduz LR a cada 30 epochs
```

#### ReduceLROnPlateau:
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(otimizador, 'min', patience=10)

# No loop de treinamento:
for epoch in range(EPOCHS):
    # ... treinamento ...
    scheduler.step(perda_validacao)  # Reduz LR se perda não melhora
```

### 🎯 Diferentes Funções de Perda

#### CrossEntropy (ao invés de NLLLoss):
```python
# Modificar modelo para não usar LogSoftmax
def forward(self, X):
    X = F.relu(self.linear1(X))
    X = F.relu(self.linear2(X))
    X = self.linear3(X)  # Sem LogSoftmax
    return X

# Usar CrossEntropy
criterios = nn.CrossEntropyLoss()
```

#### Label Smoothing:
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
```

## 🎨 Visualização

### 📊 Personalizar Gráficos

#### Cores Diferentes:
```python
# Em visualizar_predicoes_navegacao()
cores = ['red', 'orange', 'yellow', 'green', 'blue']
cor = cores[min(int(confianca_principal // 20), 4)]  # Cor baseada na confiança
```

#### Mais Informações:
```python
titulo = f'Real: {etiqueta_real} | Pred: {predicao_principal}\n'
titulo += f'Confiança: {confianca_principal:.1f}%\n'
titulo += f'Top3: {top3_pred[0]}({top3_prob[0]*100:.1f}%) '
titulo += f'{top3_pred[1]}({top3_prob[1]*100:.1f}%) '
titulo += f'{top3_pred[2]}({top3_prob[2]*100:.1f}%)\n'
titulo += f'Perda: {perda_instantanea:.3f}'  # Adicionar perda
```

### 🔍 Visualizações Adicionais

#### Confusion Matrix:
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(modelo, dataloader, device):
    modelo.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imagens, etiquetas in dataloader:
            imagens = imagens.view(imagens.shape[0], -1)
            output = modelo(imagens.to(device))
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(etiquetas.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
```

#### Visualizar Pesos:
```python
def visualizar_pesos(modelo):
    # Visualiza pesos da primeira camada como imagens
    pesos = modelo.linear1.weight.data.cpu()
    
    fig, axes = plt.subplots(8, 16, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(min(128, len(axes))):
        peso = pesos[i].view(28, 28)
        axes[i].imshow(peso, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Neuron {i}')
    
    plt.suptitle('Pesos da Primeira Camada')
    plt.tight_layout()
    plt.show()
```

## 📊 Datasets

### 🔄 Outros Datasets

#### Fashion-MNIST:
```python
trainset = datasets.FashionMNIST('./FashionMNIST_data/', 
                                download=True, 
                                train=True, 
                                transform=transform)
```

#### CIFAR-10 (requer modificações na arquitetura):
```python
# Imagens coloridas 32x32x3
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10('./CIFAR10_data/', 
                           download=True, 
                           train=True, 
                           transform=transform)

# Modificar modelo para entrada 32*32*3 = 3072
class ModeloCIFAR(nn.Module):
    def __init__(self):
        super(ModeloCIFAR, self).__init__()
        self.linear1 = nn.Linear(32*32*3, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 10)  # 10 classes no CIFAR-10
```

### 🎨 Data Augmentation

```python
transform_augmented = transforms.Compose([
    transforms.RandomRotation(10),      # Rotação aleatória
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Translação
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalização MNIST
])

trainset = datasets.MNIST('./MNIST_data/', 
                         download=True, 
                         train=True, 
                         transform=transform_augmented)
```

## 🔬 Experimentos Avançados

### 📊 Múltiplas Arquiteturas

```python
def comparar_arquiteturas():
    arquiteturas = {
        'Simples': [784, 64, 10],
        'Média': [784, 128, 64, 10],
        'Complexa': [784, 512, 256, 128, 10]
    }
    
    resultados = {}
    
    for nome, layers in arquiteturas.items():
        modelo = criar_modelo(layers)
        precisao = treinar_modelo(modelo)
        resultados[nome] = precisao
        
    return resultados
```

### 🎯 Grid Search

```python
def grid_search():
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    
    melhores_params = None
    melhor_precisao = 0
    
    for lr in learning_rates:
        for bs in batch_sizes:
            precisao = treinar_com_params(lr, bs)
            if precisao > melhor_precisao:
                melhor_precisao = precisao
                melhores_params = {'lr': lr, 'batch_size': bs}
    
    return melhores_params, melhor_precisao
```

### 📈 Early Stopping

```python
def treino_com_early_stopping(modelo, trainloader, valloader, device, patience=5):
    melhor_perda = float('inf')
    contador_paciencia = 0
    
    for epoch in range(100):  # Máximo de epochs
        # Treino normal...
        perda_val = calcular_perda_validacao(modelo, valloader, device)
        
        if perda_val < melhor_perda:
            melhor_perda = perda_val
            contador_paciencia = 0
            # Salvar melhor modelo
            torch.save(modelo.state_dict(), 'melhor_modelo.pth')
        else:
            contador_paciencia += 1
            
        if contador_paciencia >= patience:
            print(f"Early stopping no epoch {epoch}")
            break
```

### 💾 Salvar e Carregar Modelos

```python
# Salvar modelo
def salvar_modelo(modelo, caminho):
    torch.save({
        'model_state_dict': modelo.state_dict(),
        'arquitetura': 'feedforward',
        'timestamp': time.time()
    }, caminho)

# Carregar modelo
def carregar_modelo(caminho):
    checkpoint = torch.load(caminho)
    modelo = Modelo()
    modelo.load_state_dict(checkpoint['model_state_dict'])
    return modelo
```

### 📊 Métricas Avançadas

```python
from sklearn.metrics import classification_report, f1_score

def metricas_detalhadas(modelo, dataloader, device):
    modelo.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imagens, etiquetas in dataloader:
            imagens = imagens.view(imagens.shape[0], -1)
            output = modelo(imagens.to(device))
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(etiquetas.numpy())
    
    # Relatório detalhado
    report = classification_report(all_labels, all_preds, 
                                 target_names=[f'Digit {i}' for i in range(10)])
    
    # F1-Score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return report, f1
```

---

**🎯 Use essas modificações para experimentar e melhorar o modelo conforme suas necessidades!**
