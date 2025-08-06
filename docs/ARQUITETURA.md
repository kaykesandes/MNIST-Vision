# 🏗️ Arquitetura da Rede Neural

## 📚 Índice
- [Visão Geral](#visão-geral)
- [Arquitetura Detalhada](#arquitetura-detalhada)
- [Componentes](#componentes)
- [Fluxo de Dados](#fluxo-de-dados)
- [Justificativas de Design](#justificativas-de-design)
- [Alternativas](#alternativas)

## 🎯 Visão Geral

A rede neural implementada é uma **Feedforward Neural Network** (também conhecida como Multi-Layer Perceptron - MLP) projetada especificamente para classificação de dígitos manuscritos do dataset MNIST.

## 🏗️ Arquitetura Detalhada

### 📊 Estrutura Geral:
```
INPUT LAYER    HIDDEN LAYER 1    HIDDEN LAYER 2    OUTPUT LAYER
   (784)     →     (128)      →      (64)       →     (10)
   
28×28 pixels  →  128 neurons   →   64 neurons   →  10 classes
               +ReLU activation  +ReLU activation  +LogSoftmax
```

### 🔢 Especificações Numéricas:

| Camada | Entrada | Saída | Parâmetros | Ativação |
|--------|---------|-------|------------|----------|
| Linear1 | 784 | 128 | 100,352 | ReLU |
| Linear2 | 128 | 64 | 8,192 | ReLU |
| Linear3 | 64 | 10 | 640 | LogSoftmax |
| **Total** | - | - | **109,184** | - |

### 📐 Cálculo de Parâmetros:

#### Camada 1 (Linear1):
```
Pesos: 784 × 128 = 100,352
Bias: 128
Total: 100,352 + 128 = 100,480
```

#### Camada 2 (Linear2):
```
Pesos: 128 × 64 = 8,192  
Bias: 64
Total: 8,192 + 64 = 8,256
```

#### Camada 3 (Linear3):
```
Pesos: 64 × 10 = 640
Bias: 10  
Total: 640 + 10 = 650
```

#### Total de Parâmetros:
```
100,480 + 8,256 + 650 = 109,386 parâmetros treináveis
```

## 🧩 Componentes

### 1. 📥 Camada de Entrada (Input Layer)

**Dimensão:** 784 neurônios

**Justificativa:** 
- Imagens MNIST são 28×28 pixels = 784 pixels
- Cada pixel representa um valor de intensidade (0-1)
- Achatamento da matriz 2D para vetor 1D

```python
# Preprocessamento da entrada
imagens = imagens.view(imagens.shape[0], -1)  # [batch, 28, 28] → [batch, 784]
```

### 2. 🧠 Primeira Camada Oculta (Hidden Layer 1)

**Dimensão:** 128 neurônios  
**Ativação:** ReLU (Rectified Linear Unit)

**Características:**
- **Função:** Extração de características de baixo nível
- **ReLU:** `f(x) = max(0, x)` - introduz não-linearidade
- **Vantagens do ReLU:**
  - Computacionalmente eficiente
  - Evita problema do vanishing gradient
  - Esparsidade (muitos zeros)

```python
x = F.relu(self.linear1(x))  # Aplica transformação linear + ReLU
```

### 3. 🧠 Segunda Camada Oculta (Hidden Layer 2)

**Dimensão:** 64 neurônios
**Ativação:** ReLU

**Características:**
- **Função:** Combinação de características de alto nível
- **Redução dimensional:** 128 → 64 (compressão de informação)
- **Representação abstrata:** Padrões mais complexos

### 4. 📤 Camada de Saída (Output Layer)

**Dimensão:** 10 neurônios (um para cada dígito 0-9)
**Ativação:** LogSoftmax

**Características:**
- **LogSoftmax:** Converte logits em log-probabilidades
- **Estabilidade numérica:** Evita overflow/underflow
- **Compatibilidade:** Funciona com NLLLoss

```python
# LogSoftmax aplicado na dimensão das classes
return F.log_softmax(x, dim=1)
```

## 📊 Fluxo de Dados

### 1. **Entrada → Camada 1:**
```
Input: [batch_size, 784]
Transformação: x₁ = ReLU(W₁x + b₁)
Output: [batch_size, 128]
```

### 2. **Camada 1 → Camada 2:**
```
Input: [batch_size, 128]  
Transformação: x₂ = ReLU(W₂x₁ + b₂)
Output: [batch_size, 64]
```

### 3. **Camada 2 → Saída:**
```
Input: [batch_size, 64]
Transformação: x₃ = W₃x₂ + b₃
Output: [batch_size, 10]
```

### 4. **Ativação Final:**
```
Input: [batch_size, 10]
Transformação: output = LogSoftmax(x₃)
Output: [batch_size, 10] (log-probabilidades)
```

## 🎯 Justificativas de Design

### 1. **Por que Feedforward?**
- ✅ **Simplicidade:** Fácil de implementar e entender
- ✅ **Eficiência:** Computacionalmente leve
- ✅ **Adequação:** MNIST é um problema relativamente simples
- ✅ **Baseline:** Boa referência para comparações

### 2. **Por que 128 → 64 neurônios?**
- 🎯 **Redução gradual:** Evita perda brusca de informação
- 🎯 **Capacidade adequada:** Suficiente para MNIST
- 🎯 **Eficiência:** Não é excessivamente grande
- 🎯 **Experiência empírica:** Funciona bem na prática

### 3. **Por que ReLU?**
- ⚡ **Velocidade:** Computacionalmente eficiente
- ⚡ **Gradientes:** Não sofre vanishing gradient
- ⚡ **Esparsidade:** Ativa apenas neurônios relevantes
- ⚡ **Simplicidade:** Fácil de implementar

### 4. **Por que LogSoftmax + NLLLoss?**
- 🔢 **Estabilidade numérica:** Evita problemas com exp()
- 🔢 **Eficiência:** Combinação otimizada
- 🔢 **Gradientes:** Melhores propriedades de otimização

## 🔄 Alternativas Consideradas

### 1. **Arquiteturas Alternativas:**

#### Mais Simples:
```
784 → 64 → 10
```
**Pros:** Mais rápida, menos parâmetros
**Cons:** Menor capacidade de representação

#### Mais Complexa:
```
784 → 256 → 128 → 64 → 10
```
**Pros:** Maior capacidade
**Cons:** Overfitting potencial, mais lenta

#### CNN (Convolutional):
```
28×28 → Conv2D → MaxPool → Conv2D → FC → 10
```
**Pros:** Melhor para imagens, invariância espacial
**Cons:** Mais complexa, desnecessária para MNIST

### 2. **Funções de Ativação Alternativas:**

#### Sigmoid:
```python
x = torch.sigmoid(self.linear1(x))
```
**Pros:** Saída limitada (0,1)
**Cons:** Vanishing gradient, computacionalmente cara

#### Tanh:
```python
x = torch.tanh(self.linear1(x))
```
**Pros:** Saída centrada em zero (-1,1)
**Cons:** Vanishing gradient

#### Leaky ReLU:
```python
x = F.leaky_relu(self.linear1(x), negative_slope=0.01)
```
**Pros:** Evita "dead neurons"
**Cons:** Mais complexa

### 3. **Funções de Perda Alternativas:**

#### CrossEntropy:
```python
criterio = nn.CrossEntropyLoss()
# Requer modificação na saída (sem LogSoftmax)
```

#### MSE (Mean Squared Error):
```python
criterio = nn.MSELoss()
# Inadequada para classificação
```

## 📊 Análise de Complexidade

### **Complexidade Temporal:**
- **Forward Pass:** O(n × m) onde n = batch_size, m = parâmetros
- **Backward Pass:** O(n × m)
- **Total por batch:** O(2nm)

### **Complexidade Espacial:**
- **Parâmetros:** 109,386 × 4 bytes = ~437KB
- **Ativações:** batch_size × (784 + 128 + 64 + 10) × 4 bytes
- **Gradientes:** Mesmo tamanho dos parâmetros

### **Performance Esperada:**
```
Dataset: 60,000 imagens de treino
Batch Size: 64
Batches por epoch: 938
Tempo por epoch: ~30-60s (CPU)
Precisão esperada: 90%+
```

## 🎨 Representação Visual

### Arquitetura Completa:
```
INPUT LAYER (784)
    |
    | W₁[784×128] + b₁[128]
    ↓
HIDDEN LAYER 1 (128)
    |
    | ReLU activation
    | W₂[128×64] + b₂[64]
    ↓
HIDDEN LAYER 2 (64)
    |
    | ReLU activation  
    | W₃[64×10] + b₃[10]
    ↓
OUTPUT LAYER (10)
    |
    | LogSoftmax
    ↓
LOG-PROBABILITIES
```

### Fluxo de Gradientes (Backpropagation):
```
Loss ← NLLLoss
  ↑
LogSoftmax ← Output Layer
  ↑
ReLU ← Hidden Layer 2
  ↑  
ReLU ← Hidden Layer 1
  ↑
Input Layer
```

## 🔧 Configurações de Treinamento

### Otimizador:
```python
optimizer = optim.SGD(
    modelo.parameters(),
    lr=0.01,        # Taxa de aprendizado
    momentum=0.5    # Momentum para acelerar convergência
)
```

### Função de Perda:
```python
criterion = nn.NLLLoss()  # Negative Log Likelihood
```

### Hyperparâmetros:
```python
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.01
MOMENTUM = 0.5
```

---

**🎯 Esta arquitetura foi escolhida por equilibrar simplicidade, eficiência e performance adequada para o problema MNIST.**
