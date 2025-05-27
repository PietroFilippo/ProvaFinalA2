import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def aplicar_pre_processamento(imagem_original):
    # Redimensionamento para 128x128
    imagem_redimensionada = cv2.resize(imagem_original, (128, 128))
    
    # Aplicação do filtro Gaussiano
    imagem_suavizada = cv2.GaussianBlur(imagem_redimensionada, (5, 5), 0)
    
    # Conversão para tons de cinza
    imagem_cinza = cv2.cvtColor(imagem_suavizada, cv2.COLOR_BGR2GRAY)
    
    # Equalização de histograma
    imagem_equalizada = cv2.equalizeHist(imagem_cinza)
    
    return {
        'original': imagem_original,
        'redimensionada': imagem_redimensionada,
        'suavizada': imagem_suavizada,
        'cinza': imagem_cinza,
        'equalizada': imagem_equalizada
    }

def exibir_etapas_pre_processamento(etapas, titulo):
    plt.figure(figsize=(15, 10))
    
    # Imagem original
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(etapas['original'], cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    # Imagem redimensionada
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(etapas['redimensionada'], cv2.COLOR_BGR2RGB))
    plt.title('Redimensionada (128x128)')
    plt.axis('off')
    
    # Imagem após filtro Gaussiano
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(etapas['suavizada'], cv2.COLOR_BGR2RGB))
    plt.title('Após Filtro Gaussiano')
    plt.axis('off')
    
    # Imagem em tons de cinza
    plt.subplot(2, 3, 4)
    plt.imshow(etapas['cinza'], cmap='gray')
    plt.title('Tons de Cinza')
    plt.axis('off')
    
    # Imagem após equalização
    plt.subplot(2, 3, 5)
    plt.imshow(etapas['equalizada'], cmap='gray')
    plt.title('Após Equalização')
    plt.axis('off')
    
    # Histograma da imagem equalizada
    plt.subplot(2, 3, 6)
    plt.hist(etapas['equalizada'].flatten(), bins=256, range=[0, 256])
    plt.title('Histograma Equalizado')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')
    
    plt.suptitle(titulo, fontsize=16)
    plt.tight_layout()
    return plt

def processar_imagens(diretorio_base):
    categorias = ['gatos', 'cachorros']
    
    for categoria in categorias:
        diretorio = os.path.join(diretorio_base, categoria)
        if not os.path.exists(diretorio):
            print(f"Diretório {diretorio} não encontrado.")
            continue
            
        # Cria um diretório para salvar as imagens processadas
        diretorio_saida = os.path.join('imagens_processadas', categoria)
        os.makedirs(diretorio_saida, exist_ok=True)
        
        # Processa cada imagem
        for arquivo in os.listdir(diretorio):
            if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                caminho = os.path.join(diretorio, arquivo)
                
                # Carrega as imagem
                imagem = cv2.imread(caminho)
                if imagem is None:
                    print(f"Não foi possível carregar {caminho}")
                    continue
                
                # Aplica o pré-processamento
                etapas = aplicar_pre_processamento(imagem)
                
                # Exibe e salva os resultados
                plt = exibir_etapas_pre_processamento(etapas, f"Pré-processamento: {categoria}/{arquivo}")
                plt.savefig(os.path.join(diretorio_saida, f"processado_{arquivo}.png"))
                plt.close()
                
                # Salva a imagem equalizada
                cv2.imwrite(
                    os.path.join(diretorio_saida, f"equalizada_{arquivo}"),
                    etapas['equalizada']
                )
                
                print(f"Processamento concluído para {caminho}")

def main():
    print("Demonstração de pré-processamento de imagens")
    
    os.makedirs('imagens_processadas', exist_ok=True)
    
    # Processa todas as imagens
    processar_imagens('imagens')
    
    print("\nTodas as imagens foram processadas e salvas em 'imagens_processadas/'")
    print("Para visualizar o processo de pré-processamento, verifique as imagens salvas.")

if __name__ == "__main__":
    main() 