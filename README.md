# Link do Projeto:
(https://colab.research.google.com/drive/1G1bYMqgHySTBupL3LFSdbrB8uXosiKhp?usp=sharing)

# Segmentação de Vasos Retinianos com U-Net e U-Net++

Este projeto utiliza redes neurais convolucionais U-Net e U-Net++ para segmentação de vasos sanguíneos em imagens da retina a partir do **DRIVE Dataset**. 
O principal objetivo é utlizar o modelo unet++ para melhor precisão da retina em relação ao modelo unet convencional.

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
- Binarização das máscaras.
- Extração de patches (128x128 com stride 64).
- Aumento de dados com `ImageDataGenerator`.

## 🧠 Modelos

### U-Net

Modelo de segmentação com arquitetura encoder-decoder com skip connections.

### U-Net++

Extensão da U-Net que introduz conexões densas entre os blocos de decodificação, melhorando o fluxo de informações e a segmentação.

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

## 📦 Resultados Salvos

- `unetpp_drive_best.h5`: Melhor modelo da U-Net++ com base na `val_loss`
- `unet_best_drive.h5`: Melhor modelo da U-Net convencional

---

## 📌 Observação

O `threshold=0.1` foi utilizado para binarizar as predições. Isso ajuda a captar vasos mais finos, mas deve ser ajustado conforme necessário. Um valor muito baixo pode aumentar o número de falsos positivos.

---

## 📎 Referência

- Dataset: [DRIVE - Digital Retinal Images for Vessel Extraction](https://drive.grand-challenge.org/)
- Artigos:
  - U-Net: *Ronneberger et al., 2015*
  - U-Net++: *Zhou et al., 2018*
