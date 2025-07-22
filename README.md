# Link do Projeto:
https://colab.research.google.com/drive/1G1bYMqgHySTBupL3LFSdbrB8uXosiKhp?usp=sharing

# Segmentação de Vasos Retinianos com U-Net e U-Net++

Este projeto utiliza redes neurais convolucionais U-Net e U-Net++ para segmentação de vasos sanguíneos em imagens da retina a partir do **DRIVE Dataset**. 

O objetivo é identificar estruturas importantes – como vasos sanguíneos – auxiliando diagnósticos oftalmológicos. 

Visualização das iamgens:
<img width="1398" height="674" alt="image" src="https://github.com/user-attachments/assets/dba4b11d-1588-4b60-a09d-1fc73b71aaaa" />

Esse projeto é composto por 40 imagens, 20 de teste e 20 de treino, o que é pouco para um bom treinamento, por isso foi necessário gerar novos dados a partir das patches, as patchs foram essenciais para esse projeto, uma vez que as imagens originaris tinham proporções de 584x565, porem o colab tradicional limita para no máximo 128x128, nessa proporção parte da qualidade das imagens se perdiam, o que resultava em um péssimo resultado.

## 📁 Estrutura do Projeto

```
📂 DRIVE
 ├── training
 │   ├── images           # Imagens originais em .tif
 │   ├── 1st_manual       # Máscaras manuais (rótulos) em .gif
 │   └── mask             # Máscaras de campo de visão (FOV)

 ├── test
 │   ├── images          
 │   └── mask          
```

## 🚀 Tecnologias Utilizadas

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- scikit-learn
- Google Colab

## ⚙️ Pré-processamento

- Normalização das imagens (valores entre 0 e 1).
- Binarização das máscaras: A binarização é essencial para:
    Garantir que a máscara só indique presença (1.0) ou ausência (0.0) do objeto/estrutura de interesse.
    Facilitar o treino com modelos de segmentação binária, como U-Net ou SegNet.
    Evitar inconsistências nos dados vindos de arquivos de imagem.
  
- Extração de patches (128x128 com stride 64): O código divide imagens grandes em pequenos blocos (patches). Só usa os patches onde há conteúdo relevante (máscara com valores > 0), isso reduz a quantidade de dados desnecessários e melhora o desempenho do modelo de segmentação.
  
- Aumento de dados com `ImageDataGenerator`.

## 🧠 Modelos Utilizados

### 🔷 U-Net

**U-Net** é uma arquitetura de rede neural voltada para segmentação semântica pixel a pixel. Ela é composta por:

- **Encoder (contrator):** extrai características com camadas de convolução, ReLU e MaxPooling.
- **Decoder (expansor):** reconstrói a imagem com camadas de transposed convolution.
- **Skip connections:** liga diretamente camadas correspondentes do encoder ao decoder, preservando detalhes espaciais.

> 🔎 Útil para tarefas com imagens médicas e dados limitados, oferece bons resultados com custo computacional moderado.

---

### 🔶 U-Net++

**U-Net++** é uma extensão da U-Net, com melhorias significativas para segmentações mais complexas.

#### Principais Diferenças:

- **Conexões aninhadas e densas:** conecta múltiplos níveis intermediários entre encoder e decoder.
- **Maior profundidade:** incorpora mais camadas para melhorar a extração semântica.
- **Deep supervision (opcional):** supervisiona saídas intermediárias durante o treinamento.

> 🚀 Ideal para quando se busca maior precisão, mesmo com maior custo computacional.

---

## 📊 Comparação

| Característica             | U-Net                         | U-Net++                       |
|---------------------------|-------------------------------|-------------------------------|
| Estrutura                 | Encoder-decoder               | Encoder-decoder com conexões densas |
| Skip Connections          | Diretas                       | Aninhadas e profundas         |
| Aprendizado semântico     | Razoável                      | Mais refinado                 |
| Generalização             | Boa                           | Melhor em dados complexos     |
| Custo computacional       | Menor                         | Maior                         |
| Suporte a deep supervision| Não                           | Sim (opcional)                |

---

## 📈 Treinamento

- Otimizador: `Adam`
- Função de perda: `binary_crossentropy`
- Métricas: `accuracy`, além de métricas personalizadas com base em segmentação binária
- EarlyStopping e ModelCheckpoint utilizados para evitar overfitting.

## 📊 Métricas Avaliadas

As métricas utilizadas para comparar os modelos foram:

- **Loss**
- **Acurácia**
- **Sensibilidade (Recall)**
- **Especificidade**
- **AUC-ROC**

As máscaras preditas passaram por pós-processamento usando operações morfológicas (open/close) para remoção de ruídos.

## 🔍 Visualizações

O projeto inclui:

- Visualização de imagens e máscaras originais
- Visualização de imagens aumentadas
- Comparação lado a lado entre as predições da U-Net e da U-Net++
- Exibição de predições binarizadas com limiar (threshold ajustável)

## 🧪 Avaliação e Comparação

Os modelos foram avaliados utilizando os patches de validação extraídos do conjunto original.

```python
comparar_modelos(model_unet, model, val_img_patches, val_mask_patches, threshold=0.1)
```

Este método apresenta os seguintes resultados:

- **Média da loss**
- **Média da acurácia**
- **Sensibilidade média**
- **Especificidade média**
- **AUC-ROC média**

---

## 📦 Resultados

## Unet++:

<img width="1182" height="379" alt="image" src="https://github.com/user-attachments/assets/c03a1a96-b436-4cd6-9cab-aa7bdb4424db" />

<img width="1167" height="393" alt="image" src="https://github.com/user-attachments/assets/14c8ad4d-3ea1-48fa-82c0-3b4800a86fb8" />

## Unet:
<img width="1174" height="425" alt="image" src="https://github.com/user-attachments/assets/3d5bbe00-5c2c-4999-bda2-3164ab2f1896" />

<img width="1180" height="400" alt="image" src="https://github.com/user-attachments/assets/e6dbec7b-2078-4e3d-8f5d-e95ceb0c35c5" />


## 📊 Métricas de Avaliação

### 🔹 U-Net
- **Loss média:** 0.0973  
- **Acurácia média:** 0.9622  
- **Sensibilidade média:** 0.9054  
- **Especificidade média:** 0.9185  
- **AUC-ROC média:** 0.9761  

### 🔸 U-Net++
- **Loss média:** 0.0867  
- **Acurácia média:** 0.9660  
- **Sensibilidade média:** 0.9291  
- **Especificidade média:** 0.9263  
- **AUC-ROC média:** 0.9840  

U-Net++ teve melhor desempenho em todas as métricas, indicando que:

Ela erra menos (menor loss),

Classifica com maior precisão geral (acurácia),

Tem maior sensibilidade (detecta melhor os positivos),

Maior especificidade (detecta melhor os negativos),

E melhor separação entre classes (AUC-ROC).

Portanto, U-Net++ é a melhor escolha com base nesses resultados.

---

## 📌 Observação

O `threshold=0.1` foi utilizado para binarizar as predições. Isso ajuda a captar vasos mais finos, mas deve ser ajustado conforme necessário. Um valor muito baixo pode aumentar o número de falsos positivos.

---

## 📎 Referência

- Dataset: [DRIVE - Digital Retinal Images for Vessel Extraction](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction?resource=download)
- Artigos:
  - U-Net: *Ronneberger et al., 2015*
  - U-Net++: *Zhou et al., 2018*


## 🧑‍💻 Autor

- **Phaola Paraguai Da Paixão Lustosa**
- Email: <paxaophaola@gmail.com>

---
