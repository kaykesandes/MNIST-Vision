import numpy as np
import torch 
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import random

transform = transforms.ToTensor()

trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)
        
    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        return F.log_softmax(X, dim=1)

def visualizar_predicoes_navegacao(modelo, dataloader, device, num_amostras=6, batch_idx=0):
    """Mostra predições com navegação entre batches"""
    modelo.eval()
    
    # Pega múltiplos batches para navegação
    all_batches = []
    dataiter = iter(dataloader)
    for i in range(5):  # Pega 5 batches
        try:
            batch = next(dataiter)
            all_batches.append(batch)
        except StopIteration:
            break
    
    if batch_idx >= len(all_batches):
        batch_idx = 0
        
    imagens, etiquetas = all_batches[batch_idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    with torch.no_grad():
        # Seleciona amostras aleatórias do batch atual
        indices = random.sample(range(len(imagens)), min(num_amostras, len(imagens)))
        
        for i, idx in enumerate(indices):
            imagem = imagens[idx].view(1, -1)
            etiqueta_real = etiquetas[idx].item()
            
            # Forward pass
            output = modelo(imagem.to(device))
            probabilidades = F.softmax(output, dim=1)
            
            # Pega as top 3 predições
            top3_prob, top3_pred = torch.topk(probabilidades, 3)
            top3_prob = top3_prob.cpu().numpy()[0]
            top3_pred = top3_pred.cpu().numpy()[0]
            
            predicao_principal = top3_pred[0]
            confianca_principal = top3_prob[0] * 100
            
            # Plotar imagem
            axes[i].imshow(imagens[idx].numpy().squeeze(), cmap='gray_r')
            cor = 'green' if predicao_principal == etiqueta_real else 'red'
            
            # Título com top 3 predições
            titulo = f'Real: {etiqueta_real} | Pred: {predicao_principal}\n'
            titulo += f'Top3: {top3_pred[0]}({top3_prob[0]*100:.1f}%) '
            titulo += f'{top3_pred[1]}({top3_prob[1]*100:.1f}%) '
            titulo += f'{top3_pred[2]}({top3_prob[2]*100:.1f}%)'
            
            axes[i].set_title(titulo, color=cor, fontweight='bold', fontsize=9)
            axes[i].axis('off')
    
    plt.suptitle(f'Batch {batch_idx+1}/{len(all_batches)} - Forward Pass Predictions', fontsize=14)
    plt.tight_layout()
    plt.show()
    modelo.train()
    
    return len(all_batches)

def navegacao_interativa(modelo, dataloader, device):
    """Sistema de navegação interativa entre predições"""
    print("\n🔍 === NAVEGAÇÃO INTERATIVA DE PREDIÇÕES ===")
    print("Comandos: 'next' (próximo), 'prev' (anterior), 'quit' (sair)")
    
    batch_atual = 0
    total_batches = 0
    
    while True:
        print(f"\n📊 Visualizando batch {batch_atual + 1}...")
        total_batches = visualizar_predicoes_navegacao(modelo, dataloader, device, 6, batch_atual)
        
        comando = input(f"\nComando (next/prev/quit) [Batch {batch_atual+1}/{total_batches}]: ").strip().lower()
        
        if comando in ['next', 'n']:
            batch_atual = (batch_atual + 1) % total_batches
            print("➡️ Próximo batch...")
        elif comando in ['prev', 'p']:
            batch_atual = (batch_atual - 1) % total_batches
            print("⬅️ Batch anterior...")
        elif comando in ['quit', 'q', 'exit']:
            print("👋 Saindo da navegação...")
            break
        else:
            print("❌ Comando inválido! Use: next, prev, ou quit")

def forward_nextview(modelo, dataloader, device, num_sequences=3):
    """Mostra sequências de forward passes"""
    modelo.eval()
    print("\n🎬 === FORWARD NEXTVIEW - SEQUÊNCIAS DE PREDIÇÕES ===")
    
    dataiter = iter(dataloader)
    
    for seq in range(num_sequences):
        print(f"\n📺 Sequência {seq + 1}/{num_sequences}")
        
        try:
            imagens, etiquetas = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            imagens, etiquetas = next(dataiter)
        
        # Mostra 4 imagens em sequência
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        with torch.no_grad():
            for i in range(4):
                if i < len(imagens):
                    imagem = imagens[i].view(1, -1)
                    etiqueta_real = etiquetas[i].item()
                    
                    # Forward pass com detalhes
                    output = modelo(imagem.to(device))
                    probabilidades = F.softmax(output, dim=1)
                    
                    # Todas as probabilidades
                    probs = probabilidades.cpu().numpy()[0]
                    predicao = np.argmax(probs)
                    confianca = probs[predicao] * 100
                    
                    # Plot
                    axes[i].imshow(imagens[i].numpy().squeeze(), cmap='gray_r')
                    cor = 'green' if predicao == etiqueta_real else 'red'
                    
                    # Mostra probabilidades dos top 3
                    top3_idx = np.argsort(probs)[-3:][::-1]
                    titulo = f'Real: {etiqueta_real} | Pred: {predicao}\n'
                    for j, idx in enumerate(top3_idx):
                        titulo += f'{idx}:{probs[idx]*100:.1f}% '
                    
                    axes[i].set_title(titulo, color=cor, fontweight='bold')
                    axes[i].axis('off')
        
        plt.suptitle(f'Forward NextView - Sequência {seq + 1}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Pausa para visualização
        input("⏯️ Pressione Enter para próxima sequência...")
    
    modelo.train()

def treino_com_visualizacao(modelo, trainloader, valloader, device):
    otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5)
    inicio = time()
    
    criterios = nn.NLLLoss()
    EPOCHS = 3  # Reduzido para demonstração
    modelo.train()
    
    # Listas para armazenar histórico
    historico_perdas = []
    historico_precisao = []
    
    print("=== INICIANDO TREINAMENTO COM VISUALIZAÇÃO ===")
    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        perda_acumulada = 0
        batches_processados = 0
        
        for imagens, etiquetas in trainloader:
            batches_processados += 1
            
            imagens = imagens.view(imagens.shape[0], -1)
            otimizador.zero_grad()
            
            output = modelo(imagens.to(device))
            perda_instantanea = criterios(output, etiquetas.to(device))
            
            perda_instantanea.backward()
            otimizador.step()
            
            perda_acumulada += perda_instantanea.item()
            
            # Visualização a cada 300 batches
            if batches_processados % 300 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} - Batch {batches_processados}")
                print(f"  Perda atual: {perda_instantanea.item():.4f}")
                
                # Forward NextView durante o treinamento
                print("  🎬 Forward NextView atual...")
                forward_nextview(modelo, valloader, device, 1)
        
        # Calcula precisão no final de cada epoch
        precisao = calcular_precisao(modelo, valloader, device)
        perda_media = perda_acumulada / len(trainloader)
        
        historico_perdas.append(perda_media)
        historico_precisao.append(precisao)
        
        print(f"✅ Epoch {epoch+1}/{EPOCHS} - Perda: {perda_media:.4f} - Precisão: {precisao:.2f}%")
        print("-" * 60)
    
    # Gráfico do progresso do treinamento
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(historico_perdas, 'b-o')
    plt.title('Perda durante o Treinamento')
    plt.xlabel('Epoch')
    plt.ylabel('Perda')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(historico_precisao, 'g-o')
    plt.title('Precisão durante o Treinamento')
    plt.xlabel('Epoch')
    plt.ylabel('Precisão (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    tempo_total = (time() - inicio) / 60
    print(f"\n🎉 Tempo de treino: {tempo_total:.2f} minutos")
    print("=== TREINAMENTO CONCLUÍDO ===\n")

def calcular_precisao(modelo, valloader, device):
    """Calcula precisão sem prints"""
    modelo.eval()
    conta_corretas, conta_todas = 0, 0
    
    with torch.no_grad():
        for imagens, etiquetas in valloader:
            imagens = imagens.view(imagens.shape[0], -1)
            output = modelo(imagens.to(device))
            _, predicted = torch.max(output.data, 1)
            
            conta_todas += etiquetas.size(0)
            conta_corretas += (predicted.cpu() == etiquetas).sum().item()
    
    modelo.train()
    return 100 * conta_corretas / conta_todas

# Executar o treinamento
print("🚀 Inicializando modelo...")
modelo = Modelo()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)

print(f"💻 Usando device: {device}")
print(f"🧠 Modelo criado com {sum(p.numel() for p in modelo.parameters())} parâmetros")

# Treinar o modelo
treino_com_visualizacao(modelo, trainloader, valloader, device)

# Forward NextView pós-treinamento
print("\n🎬 === FORWARD NEXTVIEW PÓS-TREINAMENTO ===")
forward_nextview(modelo, valloader, device, 3)

# Navegação interativa
navegacao_interativa(modelo, valloader, device)