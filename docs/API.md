# 📚 API Reference - Referência das Funções

## 📋 Índice
- [Classe Modelo](#classe-modelo)
- [Funções de Visualização](#funções-de-visualização)
- [Funções de Treinamento](#funções-de-treinamento)
- [Utilitários](#utilitários)
- [Exemplos de Uso](#exemplos-de-uso)

## 🧠 Classe Modelo

### `class Modelo(nn.Module)`

Implementa uma rede neural feedforward para classificação MNIST.

#### **Construtor**

```python
Modelo()
```

**Descrição:** Inicializa a arquitetura da rede neural.

**Parâmetros:** Nenhum

**Atributos:**
- `linear1`: `nn.Linear(784, 128)` - Primeira camada
- `linear2`: `nn.Linear(128, 64)` - Segunda camada  
- `linear3`: `nn.Linear(64, 10)` - Camada de saída

**Exemplo:**
```python
modelo = Modelo()
print(f"Parâmetros: {sum(p.numel() for p in modelo.parameters())}")
```

#### **Método forward**

```python
forward(X: torch.Tensor) -> torch.Tensor
```

**Descrição:** Executa propagação direta na rede neural.

**Parâmetros:**
- `X` (`torch.Tensor`): Tensor de entrada com shape `[batch_size, 784]`

**Retorna:**
- `torch.Tensor`: Log-probabilidades com shape `[batch_size, 10]`

**Exemplo:**
```python
imagem = torch.randn(1, 784)
output = modelo(imagem)
probabilidades = torch.exp(output)  # Converter log-probs para probs
```

## 🎨 Funções de Visualização

### `visualizar_predicoes_navegacao()`

```python
visualizar_predicoes_navegacao(
    modelo: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_amostras: int = 6,
    batch_idx: int = 0
) -> int
```

**Descrição:** Cria visualização interativa das predições com capacidade de navegação entre batches.

**Parâmetros:**
- `modelo` (`nn.Module`): Modelo neural treinado
- `dataloader` (`DataLoader`): Carregador de dados para visualização
- `device` (`torch.device`): Dispositivo de processamento (CPU/GPU)
- `num_amostras` (`int`, opcional): Número de imagens a mostrar (padrão: 6)
- `batch_idx` (`int`, opcional): Índice do batch a visualizar (padrão: 0)

**Retorna:**
- `int`: Número total de batches disponíveis para navegação

**Exceções:**
- `IndexError`: Se `batch_idx` for inválido (é tratado automaticamente)

**Exemplo:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_batches = visualizar_predicoes_navegacao(modelo, valloader, device, 8, 2)
print(f"Total de batches disponíveis: {total_batches}")
```

### `forward_nextview()`

```python
forward_nextview(
    modelo: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_sequences: int = 3
) -> None
```

**Descrição:** Mostra sequências detalhadas de forward passes com análise de probabilidades.

**Parâmetros:**
- `modelo` (`nn.Module`): Modelo neural para análise
- `dataloader` (`DataLoader`): Fonte de dados
- `device` (`torch.device`): Dispositivo de processamento
- `num_sequences` (`int`, opcional): Número de sequências a mostrar (padrão: 3)

**Retorna:** `None`

**Comportamento:**
- Mostra 4 imagens por sequência
- Pausa entre sequências aguardando input do usuário
- Exibe top-3 predições com probabilidades

**Exemplo:**
```python
# Demonstração com 5 sequências
forward_nextview(modelo, trainloader, device, 5)
```

### `navegacao_interativa()`

```python
navegacao_interativa(
    modelo: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> None
```

**Descrição:** Interface de linha de comando para navegação interativa entre predições.

**Parâmetros:**
- `modelo` (`nn.Module`): Modelo neural para predições
- `dataloader` (`DataLoader`): Dados para visualização
- `device` (`torch.device`): Dispositivo de processamento

**Retorna:** `None`

**Comandos Aceitos:**
- `'next'`, `'n'`: Próximo batch
- `'prev'`, `'p'`: Batch anterior
- `'quit'`, `'q'`, `'exit'`: Sair da navegação

**Exemplo:**
```python
# Inicia navegação interativa
navegacao_interativa(modelo, valloader, device)
```

## 🎓 Funções de Treinamento

### `treino_com_visualizacao()`

```python
treino_com_visualizacao(
    modelo: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    device: torch.device
) -> None
```

**Descrição:** Executa treinamento completo do modelo com visualização em tempo real.

**Parâmetros:**
- `modelo` (`nn.Module`): Modelo a ser treinado
- `trainloader` (`DataLoader`): Dados de treinamento
- `valloader` (`DataLoader`): Dados de validação
- `device` (`torch.device`): Dispositivo de processamento

**Retorna:** `None`

**Configuração Interna:**
```python
# Otimizador
otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5)

# Função de perda
criterios = nn.NLLLoss()

# Hyperparâmetros
EPOCHS = 3
```

**Visualizações Geradas:**
- Progress durante treinamento (a cada 300 batches)
- Gráficos de perda e precisão por epoch
- Forward NextView durante o processo

**Exemplo:**
```python
# Treinar modelo com visualização completa
treino_com_visualizacao(modelo, trainloader, valloader, device)
```

### `calcular_precisao()`

```python
calcular_precisao(
    modelo: nn.Module,
    valloader: DataLoader,
    device: torch.device
) -> float
```

**Descrição:** Calcula precisão do modelo no dataset de validação.

**Parâmetros:**
- `modelo` (`nn.Module`): Modelo a ser avaliado
- `valloader` (`DataLoader`): Dados de validação
- `device` (`torch.device`): Dispositivo de processamento

**Retorna:**
- `float`: Precisão como porcentagem (0-100)

**Fórmula:**
```
Precisão = (Predições Corretas / Total de Amostras) × 100
```

**Comportamento:**
- Coloca modelo em modo de avaliação (`model.eval()`)
- Desabilita gradientes com `torch.no_grad()`
- Restaura modo de treinamento ao final

**Exemplo:**
```python
precisao = calcular_precisao(modelo, valloader, device)
print(f"Precisão do modelo: {precisao:.2f}%")
```

## 🛠️ Utilitários

### Configuração do Dataset

```python
# Transformação padrão
transform = transforms.ToTensor()

# Dataset de treino
trainset = datasets.MNIST(
    './MNIST_data/',
    download=True,
    train=True,
    transform=transform
)

# Dataset de validação
valset = datasets.MNIST(
    './MNIST_data/',
    download=True,
    train=False,
    transform=transform
)
```

### Configuração de DataLoaders

```python
# DataLoader de treino
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

# DataLoader de validação
valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=64,
    shuffle=True
)
```

### Configuração de Device

```python
# Detecção automática CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mover modelo para device
modelo.to(device)
```

## 📝 Exemplos de Uso

### **Exemplo 1: Uso Básico Completo**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Configuração
transform = transforms.ToTensor()
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# 2. Modelo e device
modelo = Modelo()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)

# 3. Treinamento
treino_com_visualizacao(modelo, trainloader, valloader, device)

# 4. Avaliação
precisao = calcular_precisao(modelo, valloader, device)
print(f"Precisão final: {precisao:.2f}%")

# 5. Visualização interativa
navegacao_interativa(modelo, valloader, device)
```

### **Exemplo 2: Predição em Nova Imagem**

```python
# Carregar imagem personalizada
def predizer_imagem(modelo, imagem_path, device):
    from PIL import Image
    
    # Carregar e preprocessar
    imagem = Image.open(imagem_path).convert('L')  # Escala de cinza
    imagem = imagem.resize((28, 28))
    transform = transforms.ToTensor()
    tensor_imagem = transform(imagem).unsqueeze(0)  # Adicionar batch dimension
    
    # Predição
    modelo.eval()
    with torch.no_grad():
        tensor_imagem = tensor_imagem.view(1, -1)  # Achatar
        output = modelo(tensor_imagem.to(device))
        probabilidades = torch.exp(output)
        predicao = torch.argmax(probabilidades, dim=1)
        confianca = torch.max(probabilidades)
    
    return predicao.item(), confianca.item()

# Usar
digito, confianca = predizer_imagem(modelo, 'meu_digito.png', device)
print(f"Predição: {digito} (confiança: {confianca*100:.1f}%)")
```

### **Exemplo 3: Avaliação Detalhada**

```python
def avaliacao_completa(modelo, dataloader, device):
    """Avaliação detalhada com métricas por classe"""
    modelo.eval()
    
    # Coletar todas as predições
    todas_predicoes = []
    todas_etiquetas = []
    
    with torch.no_grad():
        for imagens, etiquetas in dataloader:
            imagens = imagens.view(imagens.shape[0], -1)
            output = modelo(imagens.to(device))
            _, predicoes = torch.max(output, 1)
            
            todas_predicoes.extend(predicoes.cpu().numpy())
            todas_etiquetas.extend(etiquetas.numpy())
    
    # Calcular precisão por classe
    from collections import defaultdict
    precisao_por_classe = defaultdict(lambda: {'corretas': 0, 'total': 0})
    
    for pred, real in zip(todas_predicoes, todas_etiquetas):
        precisao_por_classe[real]['total'] += 1
        if pred == real:
            precisao_por_classe[real]['corretas'] += 1
    
    # Relatório
    print("=== RELATÓRIO DE AVALIAÇÃO ===")
    for classe in range(10):
        corretas = precisao_por_classe[classe]['corretas']
        total = precisao_por_classe[classe]['total']
        precisao = (corretas / total) * 100 if total > 0 else 0
        print(f"Dígito {classe}: {precisao:.1f}% ({corretas}/{total})")
    
    # Precisão geral
    total_corretas = sum(precisao_por_classe[i]['corretas'] for i in range(10))
    total_amostras = sum(precisao_por_classe[i]['total'] for i in range(10))
    precisao_geral = (total_corretas / total_amostras) * 100
    print(f"\nPrecisão Geral: {precisao_geral:.2f}%")

# Usar
avaliacao_completa(modelo, valloader, device)
```

### **Exemplo 4: Salvar e Carregar Modelo**

```python
# Salvar modelo treinado
def salvar_modelo(modelo, caminho='modelo_mnist.pth'):
    torch.save({
        'model_state_dict': modelo.state_dict(),
        'arquitetura': 'feedforward_784_128_64_10',
        'precisao': calcular_precisao(modelo, valloader, device)
    }, caminho)
    print(f"Modelo salvo em {caminho}")

# Carregar modelo
def carregar_modelo(caminho='modelo_mnist.pth'):
    checkpoint = torch.load(caminho)
    modelo = Modelo()
    modelo.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modelo carregado. Precisão anterior: {checkpoint['precisao']:.2f}%")
    return modelo

# Usar
salvar_modelo(modelo)
modelo_carregado = carregar_modelo()
```

## 🔧 Configurações Avançadas

### **Personalizar Hyperparâmetros**

```python
def treino_personalizado(modelo, trainloader, valloader, device, config):
    """Treinamento com configurações personalizadas"""
    
    # Configurações
    lr = config.get('learning_rate', 0.01)
    momentum = config.get('momentum', 0.5)
    epochs = config.get('epochs', 3)
    
    # Otimizador
    otimizador = optim.SGD(modelo.parameters(), lr=lr, momentum=momentum)
    criterios = nn.NLLLoss()
    
    # Treinamento
    for epoch in range(epochs):
        for imagens, etiquetas in trainloader:
            imagens = imagens.view(imagens.shape[0], -1)
            otimizador.zero_grad()
            
            output = modelo(imagens.to(device))
            loss = criterios(output, etiquetas.to(device))
            
            loss.backward()
            otimizador.step()

# Usar
config = {
    'learning_rate': 0.1,
    'momentum': 0.9,
    'epochs': 5
}
treino_personalizado(modelo, trainloader, valloader, device, config)
```

---

**📚 Esta API reference fornece todas as informações necessárias para usar e estender o projeto MNIST!**
