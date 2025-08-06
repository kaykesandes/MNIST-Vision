# 🧠 MNIST Neural Network - Classificação de Dígitos com Visualização Interativa

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🎯 Rede neural feedforward para classificação de dígitos manuscritos do dataset MNIST com visualização interativa em tempo real do processo de treinamento e predições.

## 📋 Índice
- [Visão Geral](#-visão-geral)
- [Características](#-características)
- [Instalação](#️-instalação)
- [Uso](#-uso)
- [Arquitetura](#️-arquitetura)
- [Resultados](#-resultados)
- [Exemplos](#-exemplos)
- [Contribuindo](#-contribuindo)
- [Documentação](#-documentação)

## 🎯 Visão Geral

Este projeto implementa uma rede neural feedforward completa para classificação de dígitos manuscritos (0-9) usando o famoso dataset MNIST. O diferencial está na **visualização interativa** que permite acompanhar o modelo "aprendendo" em tempo real.

### 🎨 Principais Funcionalidades:
- ✅ **Treinamento automatizado** com PyTorch
- 📊 **Visualização em tempo real** das predições durante o treino
- 🎬 **Forward NextView** - Sequências detalhadas de predições
- 🔍 **Navegação interativa** entre diferentes batches de dados
- 📈 **Gráficos de progresso** (perda e precisão ao longo dos epochs)
- 🎯 **Top-3 predições** com probabilidades detalhadas
- 🎨 **Interface colorida** (verde para acertos, vermelho para erros)

## 🚀 Características

| Característica | Descrição |
|----------------|-----------|
| **Framework** | PyTorch 2.0+ |
| **Dataset** | MNIST (70.000 imagens de dígitos manuscritos) |
| **Arquitetura** | Feedforward Neural Network (784→128→64→10) |
| **Otimizador** | SGD com momentum (lr=0.01, momentum=0.5) |
| **Função de Perda** | Negative Log Likelihood Loss |
| **Visualização** | Matplotlib com interface interativa |
| **Device Support** | CPU/GPU (detecção automática) |
| **Batch Size** | 64 imagens por batch |

## 🛠️ Instalação

### Pré-requisitos:
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passo a passo:

1. **Clone o repositório:**
```bash
git clone [seu-repositorio]
cd mnist-neural-network
```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

**Ou instale manualmente:**
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.21.0 matplotlib>=3.5.0
```

3. **Execute o programa:**
```bash
python main.py
```

## 📖 Uso

### Execução Básica:
```bash
python main.py
```

### 🔄 Fluxo de Execução:

1. **📊 Inicialização**
   - Carregamento do dataset MNIST
   - Criação da arquitetura da rede neural
   - Configuração do dispositivo (CPU/GPU)

2. **🎓 Treinamento** 
   - 3 epochs de treinamento
   - Visualização a cada 300 batches
   - Gráficos de progresso em tempo real

3. **🎬 Forward NextView**
   - Demonstração pós-treinamento
   - 3 sequências de predições detalhadas

4. **🎮 Navegação Interativa**
   - Interface para explorar predições
   - Comandos: `next`, `prev`, `quit`

### 🎮 Comandos da Interface Interativa:

| Comando | Ação |
|---------|------|
| `next` ou `n` | Avança para o próximo batch |
| `prev` ou `p` | Volta para o batch anterior |
| `quit` ou `q` | Sai da navegação |

## 🏗️ Arquitetura

### 🧠 Modelo Neural:
```
Entrada: 28x28 pixels = 784 neurônios
    ↓
Camada 1: Linear(784 → 128) + ReLU
    ↓
Camada 2: Linear(128 → 64) + ReLU  
    ↓
Camada 3: Linear(64 → 10) + LogSoftmax
    ↓
Saída: 10 classes (dígitos 0-9)
```

### 📊 Pipeline de Dados:
```
MNIST Dataset → DataLoader (batch=64) → Transformação → Modelo → Predição → Visualização
```

### ⚙️ Parâmetros do Modelo:
- **Total de parâmetros:** ~109,000
- **Camadas:** 3 camadas lineares + funções de ativação
- **Funções de ativação:** ReLU (camadas ocultas), LogSoftmax (saída)

## 📊 Resultados

### 🎯 Performance Esperada:

| Epoch | Perda Média | Precisão |
|-------|-------------|----------|
| 1 | ~1.8 | ~60% |
| 2 | ~0.8 | ~85% |
| 3 | ~0.5 | ~90%+ |

### ⏱️ Tempo de Execução:
- **CPU:** ~3-5 minutos total
- **GPU:** ~1-2 minutos total

### 📈 Métricas:
- **Precisão final:** 90%+ após 3 epochs
- **Taxa de convergência:** Rápida (visível desde o primeiro epoch)
- **Estabilidade:** Baixa variância entre execuções

## 🖼️ Exemplos

### 🎨 Visualização das Predições:
```
Real: 7 | Pred: 7
Top3: 7(95.2%) 1(3.1%) 9(1.2%)
```

### 🎬 Forward NextView:
- Mostra 4 imagens por sequência
- Probabilidades detalhadas para cada predição
- Cores indicativas (🟢 verde = acerto, 🔴 vermelho = erro)

### 📊 Gráficos de Progresso:
- **Gráfico de Perda:** Monitoramento da convergência
- **Gráfico de Precisão:** Evolução da performance

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. **Fork** o projeto
2. **Crie** uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. **Commit** suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. **Push** para a branch (`git push origin feature/nova-feature`)
5. **Abra** um Pull Request

### 💡 Ideias para Contribuições:
- Implementar outras arquiteturas (CNN, RNN)
- Adicionar métricas avançadas (F1-score, confusion matrix)
- Melhorar a interface de visualização
- Adicionar suporte para outros datasets
- Implementar early stopping
- Adicionar data augmentation

## 📚 Documentação

### 📖 Documentação Técnica:
- [Documentação do Código](docs/CODIGO.md) - Explicação detalhada de cada função
- [Guia de Arquitetura](docs/ARQUITETURA.md) - Detalhes da rede neural
- [API Reference](docs/API.md) - Referência das funções

### 🎓 Tutoriais:
- [Como Funciona](docs/COMO_FUNCIONA.md) - Conceitos de machine learning
- [Personalização](docs/PERSONALIZACAO.md) - Como modificar o modelo
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Soluções para problemas comuns

## 🐛 Troubleshooting

### Problemas Comuns:

**❓ Erro de importação do PyTorch:**
```bash
pip install torch torchvision --upgrade
```

**❓ Interface travada na navegação:**
```
Digite 'quit' e pressione Enter
```

**❓ Baixa precisão:**
```python
# Aumente o número de epochs em main.py
EPOCHS = 10  # ao invés de 3
```

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👨‍💻 Autor

**Kayke**
- 📧 Email: [seu-email]
- 💼 LinkedIn: [seu-linkedin]
- 🐙 GitHub: [seu-github]

## 🙏 Agradecimentos

- 📊 **Dataset MNIST** - Yann LeCun et al.
- 🔥 **PyTorch Team** - Framework excepcional
- 📈 **Matplotlib** - Visualizações incríveis
- 🌟 **Comunidade DIO** - Inspiração e aprendizado
- 🎓 **Comunidade de ML** - Conhecimento compartilhado

---

⭐ **Se este projeto foi útil, deixe uma estrela!**

💡 **Sugestões e feedback são sempre bem-vindos!**