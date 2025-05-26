import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import os

def carregar_imagens(diretorio, tamanho=(128, 128)):
    X = []
    y = []
    
    # Carrega as imagens de gatos
    diretorio_gatos = os.path.join(diretorio, 'gatos')
    if os.path.exists(diretorio_gatos):
        for arquivo in os.listdir(diretorio_gatos):
            if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                caminho = os.path.join(diretorio_gatos, arquivo)
                imagem = cv2.imread(caminho)
                if imagem is not None:
                    imagem = cv2.resize(imagem, tamanho)  # Redimensiona a imagem
                    X.append(imagem)
                    y.append(0)  # 0 para gatos
    
    # Carrega as imagens de cachorros
    diretorio_cachorros = os.path.join(diretorio, 'cachorros')
    if os.path.exists(diretorio_cachorros):
        for arquivo in os.listdir(diretorio_cachorros):
            if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                caminho = os.path.join(diretorio_cachorros, arquivo)
                imagem = cv2.imread(caminho)
                if imagem is not None:
                    imagem = cv2.resize(imagem, tamanho)  # Redimensiona a imagem
                    X.append(imagem)
                    y.append(1)  # 1 para cachorros
    
    return np.array(X), np.array(y)

def pre_processar_imagem(imagem):
    # Redimensiona a imagem para 128x128
    imagem_redimensionada = cv2.resize(imagem, (128, 128))
    
    # Aplica o filtro Gaussiano
    imagem_suavizada = cv2.GaussianBlur(imagem_redimensionada, (5, 5), 0)
    
    # Converte para tons de cinza
    imagem_cinza = cv2.cvtColor(imagem_suavizada, cv2.COLOR_BGR2GRAY)
    
    # Equaliza o histograma
    imagem_equalizada = cv2.equalizeHist(imagem_cinza)
    
    return imagem_equalizada

def treinar_modelo(X_train, y_train):
    # TODO: Implementar o treinamento do modelo
    pass

def avaliar_modelo(modelo, X_test, y_test):
    # TODO: Implementar a avaliação do modelo
    pass

def main():
    print("Iniciando pipeline de processamento de imagens...")
    
    # Carrega as imagens
    print("Carregando as imagens.")
    X, y = carregar_imagens('imagens')
    
    if len(X) == 0:
        print("Erro: Nenhuma imagem encontrada.")
        return
    
    # Pré-processa as imagens
    print("Pré-processando imagens...")
    X_processado = np.array([pre_processar_imagem(img) for img in X])
    
    # Reshape para o formato esperado pelo classificador
    n_amostras = len(X_processado)
    X_processado = X_processado.reshape(n_amostras, -1)
    
    print(f"Total de imagens processadas: {n_amostras}")
    print("Dimensões dos dados processados:", X_processado.shape)
    
    # TODO: Implementar o treinamento e avaliação

if __name__ == "__main__":
    main()   
