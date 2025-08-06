# 🐛 Troubleshooting - Soluções para Problemas Comuns

## 📚 Índice
- [Problemas de Instalação](#problemas-de-instalação)
- [Erros de Execução](#erros-de-execução)
- [Problemas de Performance](#problemas-de-performance)
- [Visualização](#visualização)
- [Problemas de Modelo](#problemas-de-modelo)
- [FAQ](#faq)

## 🛠️ Problemas de Instalação

### ❌ Erro: "No module named 'torch'"

**Problema:** PyTorch não está instalado.

**Soluções:**
```bash
# Opção 1: Instalação padrão
pip install torch torchvision

# Opção 2: Versão específica
pip install torch==2.0.0 torchvision==0.15.0

# Opção 3: Com CUDA (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verificação:**
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # True se GPU disponível
```

### ❌ Erro: "No module named 'matplotlib'"

**Problema:** Matplotlib não está instalado.

**Solução:**
```bash
pip install matplotlib

# Se houver problemas com backend:
pip install matplotlib pillow
```

### ❌ Problemas com Conda

**Problema:** Conflitos entre pip e conda.

**Solução:**
```bash
# Criar ambiente limpo
conda create -n mnist python=3.9
conda activate mnist

# Instalar dependências
conda install pytorch torchvision -c pytorch
conda install matplotlib numpy
```

### ❌ Erro de Permissão

**Problema:** Sem permissão para instalar pacotes.

**Soluções:**
```bash
# Opção 1: Instalar para usuário
pip install --user torch torchvision matplotlib numpy

# Opção 2: Usar ambiente virtual
python -m venv mnist_env
source mnist_env/bin/activate  # Linux/Mac
# mnist_env\Scripts\activate  # Windows
pip install torch torchvision matplotlib numpy
```

## ⚠️ Erros de Execução

### ❌ "RuntimeError: size mismatch"

**Problema:** Dimensões incompatíveis entre camadas.

**Diagnóstico:**
```python
# Adicionar prints para debug
def forward(self, X):
    print(f"Input shape: {X.shape}")
    X = F.relu(self.linear1(X))
    print(f"After linear1: {X.shape}")
    X = F.relu(self.linear2(X))
    print(f"After linear2: {X.shape}")
    X = self.linear3(X)
    print(f"After linear3: {X.shape}")
    return F.log_softmax(X, dim=1)
```

**Soluções:**
```python
# Verificar reshape da entrada
imagens = imagens.view(imagens.shape[0], -1)  # [batch, 784]

# Verificar arquitetura
self.linear1 = nn.Linear(28*28, 128)  # 784 → 128
self.linear2 = nn.Linear(128, 64)     # 128 → 64
self.linear3 = nn.Linear(64, 10)      # 64 → 10
```

### ❌ "CUDA out of memory"

**Problema:** GPU sem memória suficiente.

**Soluções:**
```python
# Reduzir batch size
trainloader = DataLoader(trainset, batch_size=32)  # ao invés de 64

# Forçar uso de CPU
device = torch.device("cpu")

# Limpar cache da GPU
torch.cuda.empty_cache()
```

### ❌ "AttributeError: '_SingleProcessDataLoaderIter' object has no attribute 'next'"

**Problema:** Uso incorreto do iterador.

**Solução:**
```python
# ❌ Errado
dataiter = iter(dataloader)
batch = dataiter.next()

# ✅ Correto
dataiter = iter(dataloader)
batch = next(dataiter)
```

### ❌ "IndexError: list index out of range"

**Problema:** Tentativa de acessar batch inexistente.

**Solução:**
```python
# Validação de índices
if batch_idx >= len(all_batches):
    batch_idx = 0

# Verificação de batches vazios
if len(all_batches) == 0:
    print("Nenhum batch disponível")
    return 0
```

### ❌ "KeyboardInterrupt" durante navegação

**Problema:** Programa travado, usuário força parada.

**Solução:**
```python
try:
    comando = input("Comando: ")
except KeyboardInterrupt:
    print("\n👋 Saindo...")
    break
```

## 📉 Problemas de Performance

### 🐌 Treinamento Muito Lento

**Diagnóstico:**
```python
import time

# Medir tempo por batch
for i, (imagens, etiquetas) in enumerate(trainloader):
    start_time = time.time()
    
    # Forward + backward pass
    output = modelo(imagens.to(device))
    loss = criterio(output, etiquetas.to(device))
    loss.backward()
    optimizer.step()
    
    batch_time = time.time() - start_time
    if i % 100 == 0:
        print(f"Batch {i}: {batch_time:.3f}s")
```

**Soluções:**
```python
# 1. Aumentar batch size
batch_size = 128  # ou mais

# 2. Usar GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Reduzir frequência de visualização
if batches_processados % 500 == 0:  # ao invés de 300

# 4. Usar DataLoader com múltiplos workers
trainloader = DataLoader(trainset, batch_size=64, 
                        shuffle=True, num_workers=4)
```

### 📊 Modelo Não Aprende (Precisão Baixa)

**Sintomas:**
- Precisão fica abaixo de 60%
- Perda não diminui
- Predições aleatórias

**Diagnóstico:**
```python
# Verificar se os dados estão corretos
for imagens, etiquetas in trainloader:
    print(f"Batch shape: {imagens.shape}")
    print(f"Labels: {etiquetas[:10]}")
    print(f"Image range: {imagens.min():.3f} to {imagens.max():.3f}")
    break

# Verificar gradientes
for name, param in modelo.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.6f}")
```

**Soluções:**
```python
# 1. Aumentar learning rate
optimizer = optim.SGD(modelo.parameters(), lr=0.1)  # ao invés de 0.01

# 2. Mais epochs
EPOCHS = 10

# 3. Verificar função de perda
criterion = nn.NLLLoss()  # Para LogSoftmax
# ou
criterion = nn.CrossEntropyLoss()  # Para logits raw

# 4. Reinicializar pesos
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

modelo.apply(init_weights)
```

### 📈 Overfitting

**Sintomas:**
- Precisão no treino >> precisão na validação
- Perda de validação aumenta

**Soluções:**
```python
# 1. Adicionar Dropout
class ModeloComDropout(nn.Module):
    def __init__(self):
        super(ModeloComDropout, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = self.dropout(X)
        X = F.relu(self.linear2(X))
        X = self.dropout(X)
        X = self.linear3(X)
        return F.log_softmax(X, dim=1)

# 2. Reduzir tamanho da rede
self.linear1 = nn.Linear(28*28, 64)   # ao invés de 128
self.linear2 = nn.Linear(64, 10)      # pular camada intermediária

# 3. Early stopping
def early_stopping(val_losses, patience=5):
    if len(val_losses) < patience:
        return False
    
    recent_losses = val_losses[-patience:]
    return all(recent_losses[i] >= recent_losses[i-1] for i in range(1, patience))
```

## 🎨 Visualização

### ❌ Gráficos Não Aparecem

**Problema:** Backend matplotlib não configurado.

**Soluções:**
```python
# 1. Configurar backend
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg'
import matplotlib.pyplot as plt

# 2. Ambiente sem interface gráfica
matplotlib.use('Agg')  # Salva em arquivo
plt.savefig('grafico.png')

# 3. Jupyter/Colab
%matplotlib inline
```

### ❌ Interface Travada

**Problema:** Loop infinito na navegação.

**Solução:**
```python
def navegacao_segura(modelo, dataloader, device, max_iterations=100):
    contador = 0
    while contador < max_iterations:
        try:
            comando = input("Comando: ").strip().lower()
            
            if comando in ['quit', 'q', 'exit']:
                break
            elif comando in ['next', 'n']:
                # lógica next
                pass
            elif comando in ['prev', 'p']:
                # lógica prev
                pass
            else:
                print("Comando inválido")
                
            contador += 1
            
        except KeyboardInterrupt:
            print("\nSaindo...")
            break
        except Exception as e:
            print(f"Erro: {e}")
            break
```

### ❌ Imagens Distorcidas

**Problema:** Normalização incorreta.

**Solução:**
```python
# Verificar range dos dados
print(f"Min: {imagens.min()}, Max: {imagens.max()}")

# Garantir range correto para imshow
imagem_plot = imagens[0].numpy().squeeze()
if imagem_plot.min() < 0:
    imagem_plot = (imagem_plot + 1) / 2  # De [-1,1] para [0,1]

plt.imshow(imagem_plot, cmap='gray')
```

## 🤖 Problemas de Modelo

### ❌ Gradientes Explodindo

**Sintomas:**
- Perda vira NaN ou inf
- Gradientes muito grandes

**Solução:**
```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)

# Ou reduzir learning rate
optimizer = optim.SGD(modelo.parameters(), lr=0.001)
```

### ❌ Gradientes Desaparecendo

**Sintomas:**
- Gradientes muito pequenos (< 1e-6)
- Modelo não aprende

**Soluções:**
```python
# 1. Mudar ativação
def forward(self, X):
    X = F.leaky_relu(self.linear1(X))  # ao invés de sigmoid
    X = F.leaky_relu(self.linear2(X))
    return F.log_softmax(self.linear3(X), dim=1)

# 2. Inicialização Xavier
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

modelo.apply(init_weights)
```

### ❌ Modelo Não Converge

**Diagnóstico:**
```python
# Plotar perda ao longo do treinamento
import matplotlib.pyplot as plt

perdas = []
for epoch in range(EPOCHS):
    for batch in trainloader:
        # treinamento...
        perdas.append(loss.item())
        
plt.plot(perdas)
plt.title('Perda ao longo do Treinamento')
plt.show()
```

**Soluções:**
```python
# 1. Learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 2. Diferentes otimizadores
optimizer = optim.Adam(modelo.parameters(), lr=0.001)

# 3. Normalização dos dados
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST stats
])
```

## ❓ FAQ

### **Q: Por que a precisão oscila muito?**
A: Normal em mini-batch learning. Use médias móveis:
```python
precisao_media = 0.9 * precisao_media + 0.1 * precisao_atual
```

### **Q: Como saber se o modelo está overfitting?**
A: Compare precisão treino vs validação:
```python
if precisao_treino - precisao_val > 10:
    print("Possível overfitting")
```

### **Q: O modelo pode memorizar os dados?**
A: Sim, especialmente com redes grandes. Use regularização:
```python
# L2 regularization
optimizer = optim.SGD(modelo.parameters(), lr=0.01, weight_decay=1e-4)
```

### **Q: Como melhorar a precisão?**
A: Várias estratégias:
1. Mais epochs
2. Data augmentation
3. Arquitetura melhor (CNN)
4. Ensemble de modelos
5. Hyperparameter tuning

### **Q: Por que usar LogSoftmax + NLLLoss?**
A: Estabilidade numérica vs Softmax + CrossEntropy.

### **Q: GPU é necessária?**
A: Não para MNIST, mas acelera muito. CPU é suficiente para aprendizado.

### **Q: Como salvar o modelo treinado?**
A:
```python
# Salvar
torch.save(modelo.state_dict(), 'modelo_mnist.pth')

# Carregar
modelo = Modelo()
modelo.load_state_dict(torch.load('modelo_mnist.pth'))
```

### **Q: O modelo funciona com outras imagens?**
A: Sim, desde que sejam:
- 28x28 pixels
- Escala de cinza
- Fundo preto, dígito branco
- Centralizadas

## 🆘 Obter Ajuda

### 📝 Como Reportar Problemas

1. **Descreva o erro:**
   - Mensagem de erro completa
   - Código que causou o erro
   - Ambiente (Python, PyTorch versions)

2. **Reproduzir o problema:**
   - Código mínimo que reproduz o erro
   - Dados de entrada usados

3. **Contexto:**
   - O que você estava tentando fazer
   - O que esperava que acontecesse
   - O que realmente aconteceu

### 🔍 Debug Sistemático

```python
def debug_modelo(modelo, dataloader, device):
    print("=== DEBUG MODELO ===")
    
    # 1. Verificar arquitetura
    print(f"Modelo: {modelo}")
    print(f"Parâmetros: {sum(p.numel() for p in modelo.parameters())}")
    
    # 2. Testar forward pass
    for imagens, etiquetas in dataloader:
        try:
            imagens = imagens.view(imagens.shape[0], -1)
            output = modelo(imagens.to(device))
            print(f"Input shape: {imagens.shape}")
            print(f"Output shape: {output.shape}")
            break
        except Exception as e:
            print(f"Erro no forward pass: {e}")
            break
    
    # 3. Verificar gradientes
    modelo.train()
    criterion = nn.NLLLoss()
    
    for imagens, etiquetas in dataloader:
        imagens = imagens.view(imagens.shape[0], -1)
        output = modelo(imagens.to(device))
        loss = criterion(output, etiquetas.to(device))
        loss.backward()
        
        for name, param in modelo.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                print(f"{name}: grad_norm = {grad_norm:.6f}")
        break
```

---

**🎯 Se o problema persistir, verifique a documentação oficial do PyTorch ou abra uma issue no repositório!**
