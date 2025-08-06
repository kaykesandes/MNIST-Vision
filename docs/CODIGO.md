# 📋 Documentação Técnica do Código

## 📚 Índice
- [Visão Geral](#visão-geral)
- [Estrutura das Classes](#estrutura-das-classes)
- [Funções Principais](#funções-principais)
- [Fluxo de Dados](#fluxo-de-dados)
- [Implementação Detalhada](#implementação-detalhada)

## 🎯 Visão Geral

Este documento fornece uma explicação técnica detalhada de cada componente do código, incluindo classes, funções, e o fluxo de execução completo.

## 🏗️ Estrutura das Classes

### `class Modelo(nn.Module)`

**Propósito:** Implementa uma rede neural feedforward para classificação de dígitos MNIST.

```python
class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)  # Camada de entrada
        self.linear2 = nn.Linear(128, 64)     # Camada oculta
        self.linear3 = nn.Linear(64, 10)      # Camada de saída
```

#### 📊 Especificações:
- **Entrada:** 784 neurônios (28×28 pixels achatados)
- **Camada 1:** 784 → 128 neurônios + ReLU
- **Camada 2:** 128 → 64 neurônios + ReLU
- **Saída:** 64 → 10 neurônios (classes 0-9) + LogSoftmax

#### ⚡ Método `forward(X)`:
```python
def forward(self, X):
    X = F.relu(self.linear1(X))    # Ativação ReLU na camada 1
    X = F.relu(self.linear2(X))    # Ativação ReLU na camada 2
    X = self.linear3(X)            # Camada final (sem ativação)
    return F.log_softmax(X, dim=1) # LogSoftmax para classificação
```

**Fluxo de Dados:**
```
Input [64, 784] → Linear1 → ReLU → [64, 128] → Linear2 → ReLU → [64, 64] → Linear3 → [64, 10] → LogSoftmax
```

## 🔧 Funções Principais

### 1. `visualizar_predicoes_navegacao()`

**Propósito:** Cria visualização interativa das predições com navegação entre batches.

```python
def visualizar_predicoes_navegacao(modelo, dataloader, device, num_amostras=6, batch_idx=0):
```

#### 📝 Parâmetros:
| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `modelo` | `nn.Module` | Rede neural treinada |
| `dataloader` | `DataLoader` | Carregador de dados |
| `device` | `torch.device` | CPU ou GPU |
| `num_amostras` | `int` | Número de imagens a mostrar (padrão: 6) |
| `batch_idx` | `int` | Índice do batch atual |

#### 🔄 Processo de Execução:

1. **Coleta de Batches:**
```python
all_batches = []
dataiter = iter(dataloader)
for i in range(5):  # Coleta 5 batches para navegação
    try:
        batch = next(dataiter)
        all_batches.append(batch)
    except StopIteration:
        break
```

2. **Seleção Aleatória:**
```python
indices = random.sample(range(len(imagens)), min(num_amostras, len(imagens)))
```

3. **Forward Pass e Análise:**
```python
output = modelo(imagem.to(device))
probabilidades = F.softmax(output, dim=1)
top3_prob, top3_pred = torch.topk(probabilidades, 3)
```

4. **Visualização:**
```python
cor = 'green' if predicao_principal == etiqueta_real else 'red'
axes[i].set_title(titulo, color=cor, fontweight='bold')
```

### 2. `navegacao_interativa()`

**Propósito:** Interface de linha de comando para navegar entre predições.

#### 🎮 Comandos Aceitos:
- `next`, `n` → Próximo batch
- `prev`, `p` → Batch anterior  
- `quit`, `q`, `exit` → Sair

#### 🔄 Loop Principal:
```python
while True:
    total_batches = visualizar_predicoes_navegacao(modelo, dataloader, device, 6, batch_atual)
    comando = input(f"\nComando (next/prev/quit) [Batch {batch_atual+1}/{total_batches}]: ")
    
    if comando in ['next', 'n']:
        batch_atual = (batch_atual + 1) % total_batches  # Navegação circular
    # ... outros comandos
```

### 3. `forward_nextview()`

**Propósito:** Demonstra sequências detalhadas de forward passes.

#### 🎬 Características:
- Mostra 4 imagens por sequência
- Análise completa de probabilidades
- Top-3 predições com percentuais
- Pausa interativa entre sequências

#### 📊 Análise de Probabilidades:
```python
probs = probabilidades.cpu().numpy()[0]  # Todas as 10 probabilidades
predicao = np.argmax(probs)              # Classe com maior probabilidade
top3_idx = np.argsort(probs)[-3:][::-1]  # Índices das 3 maiores
```

### 4. `treino_com_visualizacao()`

**Propósito:** Função principal de treinamento com feedback visual.

#### ⚙️ Configuração:
```python
otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5)
criterios = nn.NLLLoss()
EPOCHS = 3
```

#### 🔄 Loop de Treinamento:
```python
for epoch in range(EPOCHS):
    for imagens, etiquetas in trainloader:
        # 1. Preparação dos dados
        imagens = imagens.view(imagens.shape[0], -1)
        
        # 2. Forward pass
        output = modelo(imagens.to(device))
        perda_instantanea = criterios(output, etiquetas.to(device))
        
        # 3. Backward pass
        otimizador.zero_grad()
        perda_instantanea.backward()
        otimizador.step()
        
        # 4. Visualização periódica
        if batches_processados % 300 == 0:
            forward_nextview(modelo, valloader, device, 1)
```

#### 📈 Métricas Coletadas:
- **Perda por época:** `historico_perdas.append(perda_media)`
- **Precisão por época:** `historico_precisao.append(precisao)`

### 5. `calcular_precisao()`

**Propósito:** Calcula precisão do modelo no dataset de validação.

#### 📊 Fórmula:
```
Precisão = (Predições Corretas / Total de Amostras) × 100
```

#### 🔍 Implementação:
```python
with torch.no_grad():  # Desabilita gradientes para economia de memória
    for imagens, etiquetas in valloader:
        output = modelo(imagens.to(device))
        _, predicted = torch.max(output.data, 1)  # Argmax das predições
        
        conta_todas += etiquetas.size(0)
        conta_corretas += (predicted.cpu() == etiquetas).sum().item()
```

## 📊 Fluxo de Dados

### 1. **Preparação dos Dados:**
```
MNIST Raw → transforms.ToTensor() → DataLoader → Batch[64, 1, 28, 28]
```

### 2. **Preprocessamento:**
```
Batch[64, 1, 28, 28] → view(-1, 784) → Batch[64, 784]
```

### 3. **Forward Pass:**
```
Input[64, 784] → Modelo → Output[64, 10] → LogSoftmax → Probabilidades
```

### 4. **Loss Calculation:**
```
LogProbs[64, 10] + Labels[64] → NLLLoss → Scalar Loss
```

### 5. **Backward Pass:**
```
Loss.backward() → Gradientes → Optimizer.step() → Parâmetros Atualizados
```

## 🔍 Implementação Detalhada

### 🧠 Forward Pass Explicado:

#### Camada 1:
```python
# Input: [batch_size, 784]
x1 = self.linear1(x)    # [batch_size, 128]
x1 = F.relu(x1)         # Aplica ReLU: max(0, x)
```

#### Camada 2:
```python
# Input: [batch_size, 128]  
x2 = self.linear2(x1)   # [batch_size, 64]
x2 = F.relu(x2)         # Aplica ReLU
```

#### Camada de Saída:
```python
# Input: [batch_size, 64]
x3 = self.linear3(x2)           # [batch_size, 10]
output = F.log_softmax(x3, dim=1)  # Log-probabilidades
```

### 📊 Análise de Probabilidades:

#### Conversão para Probabilidades:
```python
log_probs = modelo(input)           # Log-probabilidades
probs = F.softmax(log_probs, dim=1) # Probabilidades reais (0-1)
```

#### Top-K Predições:
```python
top3_prob, top3_pred = torch.topk(probs, 3)
# top3_prob: [batch_size, 3] - probabilidades
# top3_pred: [batch_size, 3] - índices das classes
```

### 🎨 Sistema de Visualização:

#### Configuração de Plots:
```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Grid 2x3
axes = axes.ravel()  # Transforma em array 1D
```

#### Coloração Condicional:
```python
cor = 'green' if predicao == etiqueta_real else 'red'
axes[i].set_title(titulo, color=cor, fontweight='bold')
```

#### Formatação de Títulos:
```python
titulo = f'Real: {etiqueta_real} | Pred: {predicao_principal}\n'
titulo += f'Top3: {top3_pred[0]}({top3_prob[0]*100:.1f}%) '
titulo += f'{top3_pred[1]}({top3_prob[1]*100:.1f}%) '
titulo += f'{top3_pred[2]}({top3_prob[2]*100:.1f}%)'
```

## ⚡ Otimizações Implementadas

### 1. **Economia de Memória:**
```python
with torch.no_grad():  # Desabilita autograd durante avaliação
    # Código de inferência
```

### 2. **Gestão de Dispositivos:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)
dados.to(device)
```

### 3. **Processamento em Batches:**
```python
batch_size = 64  # Processa 64 imagens simultaneamente
```

### 4. **Operações Vetorizadas:**
```python
# Evita loops Python, usa operações tensor otimizadas
output = modelo(batch_input)  # Processa todo o batch de uma vez
```

## 🐛 Validações e Tratamento de Erros

### 1. **Validação de Índices:**
```python
if batch_idx >= len(all_batches):
    batch_idx = 0  # Reset para o primeiro batch
```

### 2. **Tratamento de StopIteration:**
```python
try:
    batch = next(dataiter)
except StopIteration:
    dataiter = iter(dataloader)  # Reinicia o iterador
    batch = next(dataiter)
```

### 3. **Validação de Comandos:**
```python
if comando in ['next', 'n']:
    # Comando válido
elif comando in ['prev', 'p']:
    # Comando válido  
else:
    print("❌ Comando inválido!")  # Feedback para usuário
```

## 📈 Métricas de Performance

### Complexidade Computacional:
- **Forward Pass:** O(batch_size × num_parameters)
- **Backward Pass:** O(batch_size × num_parameters)
- **Memory Usage:** ~50MB para modelo + dados

### Tempo de Execução:
- **Por Epoch:** ~30-60 segundos (CPU)
- **Forward Pass:** ~1ms por batch
- **Visualização:** ~2-3 segundos por plot

---

**📝 Nota:** Este documento detalha a implementação técnica. Para conceitos de machine learning, consulte [COMO_FUNCIONA.md](COMO_FUNCIONA.md).
