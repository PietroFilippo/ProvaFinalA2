import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import cm

def carregar_imagens_teste(diretorio, tamanho=(32, 32)):
    imagens = []
    rotulos = []
    nomes_arquivos = []
    
    # Processa imagens de gatos
    diretorio_gatos = os.path.join(diretorio, 'gatos')
    if os.path.exists(diretorio_gatos):
        for arquivo in os.listdir(diretorio_gatos):
            if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                caminho = os.path.join(diretorio_gatos, arquivo)
                imagem = cv2.imread(caminho)
                if imagem is not None:
                    # Converte de BGR para RGB
                    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
                    # Redimensiona para o tamanho esperado pelo modelo (32x32 para CIFAR-10)
                    imagem = cv2.resize(imagem, tamanho)
                    imagens.append(imagem)
                    rotulos.append(0)  # 0 para gatos
                    nomes_arquivos.append(f"gatos/{arquivo}")
    
    # Processa imagens de cachorros
    diretorio_cachorros = os.path.join(diretorio, 'cachorros')
    if os.path.exists(diretorio_cachorros):
        for arquivo in os.listdir(diretorio_cachorros):
            if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                caminho = os.path.join(diretorio_cachorros, arquivo)
                imagem = cv2.imread(caminho)
                if imagem is not None:
                    # Converte de BGR para RGB
                    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
                    # Redimensiona para o tamanho esperado pelo modelo
                    imagem = cv2.resize(imagem, tamanho)
                    imagens.append(imagem)
                    rotulos.append(1)  # 1 para cachorros
                    nomes_arquivos.append(f"cachorros/{arquivo}")
    
    return np.array(imagens), np.array(rotulos), nomes_arquivos

def pre_processar_lote(imagens):
    # Normalização (como feito durante o treinamento)
    return imagens.astype('float32') / 255.0

def exibir_resultados(imagens, rotulos_reais, predicoes, nomes_arquivos, num_mostrar=10):
    # Limita o número de imagens a exibir
    n = min(len(imagens), num_mostrar)
    
    # Seleciona índices aleatórios se houver mais imagens que o limite
    if len(imagens) > n:
        indices = np.random.choice(len(imagens), n, replace=False)
    else:
        indices = range(n)
    
    # Calcula o número de linhas e colunas para a grade
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    
    # Configura a figura com tamanho maior
    plt.figure(figsize=(15, 4 * n_rows))
    plt.subplots_adjust(hspace=0.5)
    
    # Mapeia probabilidade para cores (0=vermelho para gato, 1=azul para cachorro)
    cmap = cm.get_cmap('coolwarm')
    
    # Para cada índice selecionado
    for i, idx in enumerate(indices):
        imagem = imagens[idx]
        rotulo_real = "Gato" if rotulos_reais[idx] == 0 else "Cachorro"
        
        # Obtem a previsão
        prob = predicoes[idx][0]
        rotulo_previsto = "Gato" if prob < 0.5 else "Cachorro"
        
        # Define a cor: verde se correto, vermelho se incorreto
        cor = "green" if rotulo_real == rotulo_previsto else "red"
        
        # Plota a imagem
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(imagem)
        
        # Adiciona uma barra de confiança colorida acima da imagem
        conf_bar = np.ones((10, imagem.shape[1], 4))
        # Cor baseada na probabilidade (vermelho para gato, azul para cachorro)
        conf_bar[:, :, :3] = np.array(cmap(prob)[:3])
        
        # Insere a barra de confiança acima da imagem
        ax = plt.gca()
        ax.figure.figimage(
            conf_bar, 
            xo=ax.get_position().x0 * ax.figure.bbox.width,
            yo=(ax.get_position().y0 + ax.get_position().height) * ax.figure.bbox.height - 10,
            origin='upper'
        )
        
        # Adiciona nome do arquivo de forma mais compacta
        arquivo_nome = os.path.basename(nomes_arquivos[idx])
        
        # Título com informações
        plt.title(
            f"{arquivo_nome}\n"
            f"Real: {rotulo_real}\n"
            f"Previsto: {rotulo_previsto} ({prob:.2f})",
            color=cor, fontsize=10
        )
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('resultados_predicao.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Mostra matriz de confusão
    mostrar_matriz_confusao(rotulos_reais, predicoes)

def mostrar_matriz_confusao(rotulos_reais, predicoes):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Converte probabilidades em classes
    previsoes_classe = (predicoes > 0.5).astype(int).flatten()
    
    # Calcula a matriz de confusão
    cm = confusion_matrix(rotulos_reais, previsoes_classe)
    
    # Plota a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Gato', 'Cachorro'],
                yticklabels=['Gato', 'Cachorro'])
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png')
    plt.show()

def main():
    print("Carregando o modelo treinado...")
    try:
        modelo = load_model('modelo_classificacao_cifar10_gatos_cachorros.h5')
    except:
        print("Erro ao carregar o modelo. Verifique se o arquivo existe.")
        return
    
    print("Carregando imagens para teste...")
    imagens, rotulos, nomes_arquivos = carregar_imagens_teste('imagens')
    
    if len(imagens) == 0:
        print("Nenhuma imagem encontrada para teste.")
        return
    
    print(f"Total de imagens carregadas: {len(imagens)}")
    
    # Pré-processa as imagens
    imagens_processadas = pre_processar_lote(imagens)
    
    # Faz as previsões
    print("Realizando previsões...")
    predicoes = modelo.predict(imagens_processadas)
    
    # Calcula a acurácia
    previsoes_classe = (predicoes > 0.5).astype(int).flatten()
    acuracia = np.mean(previsoes_classe == rotulos)
    print(f"Acurácia nas imagens de teste: {acuracia:.4f}")
    
    # Exibe os resultados
    print("Exibindo resultados...")
    exibir_resultados(imagens, rotulos, predicoes, nomes_arquivos)

if __name__ == "__main__":
    main() 