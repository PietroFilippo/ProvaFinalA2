# ProvaFinalA2
Mini Projeto de Classificação de Imagens com Pré-processamento e IA

## Descrição do problema
O problema consiste na classificação binária de imagens para diferenciar gatos e cachorros. 
Esse projeto abrange o pré-processamento de imagens, o treinamento de um modelo de redes neurais convolucionais (CNN), a validação com métricas de desempenho e a predição de imagens com base no modelo treinado.

## Justificativa das técnicas utilizadas

### Modelo de classificação
- BatchNormalization: Estabiliza e acelera o treinamento que normaliza as ativações.
- Dropout (0.5): Reduz o overfitting o que força a rede a aprender características mais robustas.
- Early Stopping: Evita o sobreajuste que interrompe o treinamento quando o desempenho no conjunto de validação para de melhorar.
- ReduceLROnPlateau: Faz automaticamente o ajuste da taxa de aprendizado quando o modelo estagna.

## Etapas realizadas

1. Preparação dos dados:
   - Carregamento do dataset do CIFAR-10
   - Filtragem das classes de gatos (classe 3) e cachorros (classe 5)
   - Divisão de conjuntos de treino, validação e teste
   - Normalização das imagens (escala 0-1)

2. Pré-processamento das imagens:
   - É aplicado técnicas de processamento de imagens (redimensionamento, suavização, conversão para tons de cinza, equalização)
   - Visualização das etapas de pré-processamento para verificação dos resultados

3. Treinamento do modelo:
   - É criado uma CNN com várias camadas convolucionais
   - É feito treinamento com callbacks para otimização (early stopping, redução de taxa de aprendizado)
   - É feito o monitoramento de métricas durante o treinamento

4. Avaliação do modelo:
   - Cálculo das métricas (precisão, recall, F1-score, acurácia)
   - Geração da matriz de confusão
   - Visualização de exemplos de predições

5. Predição em novas imagens:
   - Carregamento das imagens de teste do diretório /imagens
   - Aplicação do pré-processamento nessas novas imagens
   - Predição e visualização dos resultados nessas imagens

## Resultados obtidos
O modelo CNN mostrou um desempenho bom para classificação, as três camadas convolucionais seguidas de batch normalization e max pooling foram o suficiente praa conseguir extrair as características relevantes das imagens.
Também, as técnicas como o dropout e early stopping ajudaram na redução do overfitting, que fez com que o modelo conseguisse uma boa capacidade de generalização. 
A visualização dos resultados e da matriz de confusãotambém ajudaram para identificar possíveis áreas de melhoria.

## Tempo total gasto
O desenvolvimento da prova levou 2 horas mais ou menos.

## Dificuldades encontradas
A principal dificuldade encontrada foi conseguir um bom resultado com treinamento do modelo, principalmente em balancear o trade-off da complexidade do modelo e do tempo de treinamento.