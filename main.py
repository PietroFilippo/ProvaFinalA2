import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Carrega o dataset CIFAR-10
def carregar_cifar10():
    print("Carregando o dataset CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Extraição de gatos (classe 3) e cachorros (classe 5)
    # Primeiro para treino
    cat_indices_train = np.where(y_train.flatten() == 3)[0]
    dog_indices_train = np.where(y_train.flatten() == 5)[0]
    
    x_cats_train = x_train[cat_indices_train]
    y_cats_train = np.zeros(len(cat_indices_train))  # 0 para gatos
    
    x_dogs_train = x_train[dog_indices_train]
    y_dogs_train = np.ones(len(dog_indices_train))   # 1 para cachorros
    
    # Agora para teste
    cat_indices_test = np.where(y_test.flatten() == 3)[0]
    dog_indices_test = np.where(y_test.flatten() == 5)[0]
    
    x_cats_test = x_test[cat_indices_test]
    y_cats_test = np.zeros(len(cat_indices_test))    # 0 para gatos
    
    x_dogs_test = x_test[dog_indices_test]
    y_dogs_test = np.ones(len(dog_indices_test))     # 1 para cachorros
    
    # Combina gatos e cachorros
    x_train_filtrado = np.vstack((x_cats_train, x_dogs_train))
    y_train_filtrado = np.concatenate((y_cats_train, y_dogs_train))
    
    x_test_filtrado = np.vstack((x_cats_test, x_dogs_test))
    y_test_filtrado = np.concatenate((y_cats_test, y_dogs_test))
    
    # Embaralha os dados
    indices_train = np.random.permutation(len(x_train_filtrado))
    indices_test = np.random.permutation(len(x_test_filtrado))
    
    x_train_filtrado = x_train_filtrado[indices_train]
    y_train_filtrado = y_train_filtrado[indices_train]
    x_test_filtrado = x_test_filtrado[indices_test]
    y_test_filtrado = y_test_filtrado[indices_test]
    
    print(f"Total de imagens filtradas para treino: {len(x_train_filtrado)}")
    print(f"Total de imagens filtradas para teste: {len(x_test_filtrado)}")
    
    # Contagem de classes
    classes_train, contagens_train = np.unique(y_train_filtrado, return_counts=True)
    classes_test, contagens_test = np.unique(y_test_filtrado, return_counts=True)
    
    print("\nDistribuição do conjunto de treino:")
    for classe, contagem in zip(classes_train, contagens_train):
        nome_classe = "Gato" if classe == 0 else "Cachorro"
        print(f"Classe {nome_classe}: {contagem} imagens")
    
    print("\nDistribuição do conjunto de teste:")
    for classe, contagem in zip(classes_test, contagens_test):
        nome_classe = "Gato" if classe == 0 else "Cachorro"
        print(f"Classe {nome_classe}: {contagem} imagens")
    
    return x_train_filtrado, y_train_filtrado, x_test_filtrado, y_test_filtrado

def criar_modelo_cnn(input_shape):
    modelo = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Terceira camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten e camadas densas
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout para reduzir overfitting
        layers.Dense(1, activation='sigmoid')  # Classificação binária: gato ou cachorro
    ])
    
    # Compila o modelo
    modelo.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Para classificação binária
        metrics=['accuracy']
    )
    
    return modelo

def treinar_modelo(X_train, y_train, X_val, y_val, epochs=30):
    # Cria o modelo
    input_shape = X_train.shape[1:]  # (altura, largura, canais)
    modelo = criar_modelo_cnn(input_shape)
    
    # Resumo do modelo
    modelo.summary()
    
    # Callbacks para melhorar o treinamento
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001
    )
    
    # Treina o modelo
    historico = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        batch_size=64,
        callbacks=[early_stopping, reduce_lr]
    )
    
    return modelo, historico

def avaliar_modelo(modelo, X_test, y_test):
    # Faz as predições
    y_pred_prob = modelo.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calcula as métricas
    precisao = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Imprime o relatório de classificação detalhado
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['Gato', 'Cachorro']))
    
    print(f"\nPrecisão: {precisao:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Avalia com a métrica accuracy
    _, acuracia = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"Acurácia no conjunto de teste: {acuracia:.4f}")
    
    # Mostra algumas previsões
    visualizar_predicoes(modelo, X_test, y_test)
    
    return precisao, recall, f1, acuracia

def visualizar_predicoes(modelo, X_test, y_test, num_images=5):
    # Seleciona algumas imagens aleatórias
    indices = np.random.choice(len(X_test), num_images, replace=False)
    
    plt.figure(figsize=(15, 3*num_images))
    
    for i, idx in enumerate(indices):
        imagem = X_test[idx]
        rotulo_real = "Gato" if y_test[idx] == 0 else "Cachorro"
        
        # Faz a previsão
        predicao = modelo.predict(np.expand_dims(imagem, axis=0))[0][0]
        rotulo_previsto = "Gato" if predicao < 0.5 else "Cachorro"
        
        # Adiciona a cor verde se a previsão estiver correta, vermelha se estiver errada
        cor = "green" if rotulo_real == rotulo_previsto else "red"
        
        plt.subplot(num_images, 1, i+1)
        plt.imshow(imagem)
        plt.title(f"Real: {rotulo_real} | Previsto: {rotulo_previsto} ({predicao:.4f})", color=cor)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('exemplos_predicoes.png')
    plt.show()

def visualizar_resultados(historico):
    # Plota a acurácia
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(historico.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(historico.history['val_accuracy'], label='Acurácia de Validação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.title('Acurácia do Modelo')
    
    # Plota a perda
    plt.subplot(1, 2, 2)
    plt.plot(historico.history['loss'], label='Perda de Treinamento')
    plt.plot(historico.history['val_loss'], label='Perda de Validação')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.title('Perda do Modelo')
    
    plt.tight_layout()
    plt.savefig('resultados_treinamento.png')
    plt.show()

def main():
    print("Iniciando pipeline de classificação de gatos e cachorros com CIFAR-10...")
    
    # Carrega os dados CIFAR-10 (apenas gatos e cachorros)
    X_train, y_train, X_test, y_test = carregar_cifar10()
    
    # Normaliza as imagens
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Divide em conjunto de treino e validação
    val_size = int(0.2 * len(X_train))
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]
    
    print(f"Tamanho do conjunto de treino: {X_train.shape[0]}")
    print(f"Tamanho do conjunto de validação: {X_val.shape[0]}")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")
    
    # Treina o modelo
    print("\nTreinando o modelo CNN...")
    modelo, historico = treinar_modelo(X_train, y_train, X_val, y_val, epochs=30)
    
    # Avalia o modelo
    print("\nAvaliando o modelo...")
    precisao, recall, f1, acuracia = avaliar_modelo(modelo, X_test, y_test)
    
    # Visualiza os resultados
    visualizar_resultados(historico)
    
    # Salva o modelo
    modelo.save('modelo_classificacao_cifar10_gatos_cachorros.h5')
    print("\nModelo salvo como 'modelo_classificacao_cifar10_gatos_cachorros.h5'")

if __name__ == "__main__":
    main() 