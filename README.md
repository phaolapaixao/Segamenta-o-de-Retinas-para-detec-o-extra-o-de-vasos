# Link do Projeto:
https://colab.research.google.com/drive/1G1bYMqgHySTBupL3LFSdbrB8uXosiKhp?usp=sharing

# Segmentação de Vasos Retinianos com U-Net e U-Net++

Este projeto utiliza redes neurais convolucionais U-Net e U-Net++ para segmentação de vasos sanguíneos em imagens da retina a partir do **DRIVE Dataset**. 

O objetivo é identificar estruturas importantes – como vasos sanguíneos – auxiliando diagnósticos oftalmológicos. 

### Visualização das imagens:

<img width="1398" height="674" alt="image" src="https://github.com/user-attachments/assets/dba4b11d-1588-4b60-a09d-1fc73b71aaaa" />

<img width="1027" height="498" alt="image" src="https://github.com/user-attachments/assets/d1df78e0-b25d-4573-9442-8f96317a6eac" />


Esse projeto é composto por 40 imagens, 20 de teste e 20 de treino, o que é pouco para um bom treinamento, por isso foi necessário gerar novos dados a partir das patches, as patchs foram essenciais para esse projeto, uma vez que as imagens originaris tinham proporções de 584x565, porem o colab tradicional limita para no máximo 128x128, nessa proporção parte da qualidade das imagens se perdiam, o que resultava em um péssimo resultado.

# 📁 Estrutura do Projeto

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
# 📂 Organização do Dataset - Kaggle para Google Drive

Este repositório descreve como importar datasets do **Kaggle** para o **Google Drive**, mantendo uma estrutura organizada para treinamento e teste de modelos de segmentação como U-Net e U-Net++.

---

## 📥 Como importar o dataset do Kaggle para o Google Drive

1. ### ✅ Obtenha sua API key do Kaggle:
   - Vá até [https://www.kaggle.com](https://www.kaggle.com)
   - Clique na sua foto de perfil → *Account*
   - Role até **API** e clique em **"Create New API Token"**
   - Isso fará o download de um arquivo chamado `kaggle.json`

2. ### 🔐 Faça upload da chave para o seu Colab:
   No início do notebook, execute:
   ```python
   from google.colab import files
   files.upload()  # selecione o arquivo kaggle.json
3. ### 📁 Configure o ambiente do Kaggle no Colab:
   ```python
      !mkdir -p ~/.kaggle
      !cp kaggle.json ~/.kaggle/
      !chmod 600 ~/.kaggle/kaggle.json
4. ### ⬇️ Baixe o dataset desejado:
   https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction?resource=download

   ```python
      !kaggle datasets download -d aryashah2k/drive-dataset

5. ### 📦 Extraia o conteúdo:
   ```python
      !unzip drive-dataset.zip -d /content/drive_dataset

6. ### 🔗 Monte o Google Drive:
   ```python
    from google.colab import drive
    drive.mount('/content/drive')

7. ### 📂 Organize os dados no Drive:
    Após montar o Drive, mova os arquivos extraídos:
   ```python
   !mv /content/drive_dataset /content/drive/MyDrive/DRIVE
  
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
```python
    def load_data(img_dir, mask_dir, fov_dir):
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.gif")))
    fov_files = sorted(glob.glob(os.path.join(fov_dir, "*.gif")))

    images = []
    masks = []
    fov_masks = []

    for img_path, mask_path, fov_path in zip(img_files, mask_files, fov_files):
        # Carregar imagem e normalizar
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0
        images.append(img)

        # Carregar máscara e binarizar
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.float32)
        mask[mask > 0] = 1.0 # Garantir que seja 0 ou 1
        mask = np.expand_dims(mask, axis=-1) # Adicionar canal
        masks.append(mask)

        # Carregar máscara de FOV e binarizar
        fov = Image.open(fov_path)
        fov = np.array(fov, dtype=np.float32)
        fov[fov > 0] = 1.0
        fov = np.expand_dims(fov, axis=-1)
        fov_masks.append(fov)

    return np.array(images), np.array(masks), np.array(fov_masks)
````

- Extração de patches (128x128 com stride 64): O código divide imagens grandes em pequenos blocos (patches). Só usa os patches onde há conteúdo relevante (máscara com valores > 0), isso reduz a quantidade de dados desnecessários e melhora o desempenho do modelo de segmentação.
  ```python
     def create_patches(images, masks, patch_size=128, stride=64):
    img_patches, mask_patches = [], []
    img_h, img_w = images.shape[1], images.shape[2]

    for i in range(images.shape[0]):
        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                img_patch = images[i, y:y+patch_size, x:x+patch_size]
                mask_patch = masks[i, y:y+patch_size, x:x+patch_size]
                if np.sum(mask_patch) > 0:
                    img_patches.append(img_patch)
                    mask_patches.append(mask_patch)
    return np.array(img_patches), np.array(mask_patches)
  
  X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.1, random_state=42)
  
  train_img_patches, train_mask_patches = create_patches(X_train, y_train)
  val_img_patches, val_mask_patches = create_patches(X_val, y_val)

- Aumento de dados com `ImageDataGenerator`.
  ### visulização das imagens geradas:
  
  <img width="1213" height="634" alt="image" src="https://github.com/user-attachments/assets/bdc39b8f-9006-472e-9ce9-eb9017e43e5d" />
  <img width="1211" height="651" alt="image" src="https://github.com/user-attachments/assets/6e6dbf93-9a1e-4fec-8785-c2af8454e86d" />

## 🧠 Modelos Utilizados

### 🔷 U-Net

**U-Net** é uma arquitetura de rede neural voltada para segmentação semântica pixel a pixel. Ela é composta por:

- **Encoder (contrator):** extrai características com camadas de convolução, ReLU e MaxPooling.
- **Decoder (expansor):** reconstrói a imagem com camadas de transposed convolution.
- **Skip connections:** liga diretamente camadas correspondentes do encoder ao decoder, preservando detalhes espaciais.

> 🔎 Útil para tarefas com imagens médicas e dados limitados, oferece bons resultados com custo computacional moderado.
---
```python
   def build_unet(input_shape, num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = conv_block(p4, 1024)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = safe_concat([u6, c4])
    c6 = conv_block(u6, 512)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = safe_concat([u7, c3])
    c7 = conv_block(u7, 256)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = safe_concat([u8, c2])
    c8 = conv_block(u8, 128)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = safe_concat([u9, c1])
    c9 = conv_block(u9, 64)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)

    model_unet= models.Model(inputs=[inputs], outputs=[outputs])
    return model
```
### 🔶 U-Net++

**U-Net++** é uma extensão da U-Net, com melhorias significativas para segmentações mais complexas.

### Configurações do modelo unet++
```python
import tensorflow as tf

# Função de bloco de convolução (2x Conv2D + BatchNorm + ReLU)
def conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# Concatenação segura que lida com tamanhos diferentes
def safe_concat(tensors):
    min_height = min(t.shape[1] for t in tensors)
    min_width = min(t.shape[2] for t in tensors)
    resized = [layers.Cropping2D(((0, t.shape[1] - min_height),
                                  (0, t.shape[2] - min_width)))(t)
               if t.shape[1] != min_height or t.shape[2] != min_width else t
               for t in tensors]
    return layers.Concatenate()(resized)

# Função para construir a U-Net++

def build_unet_plus_plus(input_shape, num_classes=1, deep_supervision=False):
    inputs = layers.Input(shape=input_shape)
    # Encoder
    X_0_0 = conv_block(inputs, 64)
    p0 = layers.MaxPooling2D((2, 2))(X_0_0)

    X_1_0 = conv_block(p0, 128)
    p1 = layers.MaxPooling2D((2, 2))(X_1_0)

    X_2_0 = conv_block(p1, 256)
    p2 = layers.MaxPooling2D((2, 2))(X_2_0)

    X_3_0 = conv_block(p2, 512)
    p3 = layers.MaxPooling2D((2, 2))(X_3_0)

    X_4_0 = conv_block(p3, 1024)

    # Decoder
    u3_0 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(X_4_0)
    X_3_1 = conv_block(safe_concat([X_3_0, u3_0]), 512)

    u2_0 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(X_3_0)
    X_2_1 = conv_block(safe_concat([X_2_0, u2_0]), 256)

    u2_1 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(X_3_1)
    X_2_2 = conv_block(safe_concat([X_2_0, X_2_1, u2_1]), 256)

    u1_0 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(X_2_0)
    X_1_1 = conv_block(safe_concat([X_1_0, u1_0]), 128)

    u1_1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(X_2_1)
    X_1_2 = conv_block(safe_concat([X_1_0, X_1_1, u1_1]), 128)

    u1_2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(X_2_2)
    X_1_3 = conv_block(safe_concat([X_1_0, X_1_1, X_1_2, u1_2]), 128)

    u0_0 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(X_1_0)
    X_0_1 = conv_block(safe_concat([X_0_0, u0_0]), 64)

    u0_1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(X_1_1)
    X_0_2 = conv_block(safe_concat([X_0_0, X_0_1, u0_1]), 64)

    u0_2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(X_1_2)
    X_0_3 = conv_block(safe_concat([X_0_0, X_0_1, X_0_2, u0_2]), 64)

    u0_3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(X_1_3)
    X_0_4 = conv_block(safe_concat([X_0_0, X_0_1, X_0_2, X_0_3, u0_3]), 64)

    # Deep Supervision
    if deep_supervision:
        outputs = [
            layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(X_0_1),
            layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(X_0_2),
            layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(X_0_3),
            layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(X_0_4),
        ]
        model = models.Model(inputs, outputs)
    else:
        output = layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(X_0_4)
        model = models.Model(inputs, output)

    return model
```

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

  ```python
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    checkpoint = tf.keras.callbacks.ModelCheckpoint("unetpp_drive_best.h5", save_best_only=True, monitor='val_loss')
    early = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        train_generator, # imagens geradas pela extração de patchs
        steps_per_epoch=len(train_img_patches) // BATCH_SIZE, # imagens geradas com o ImageDataGenerator
        epochs=EPOCHS,
        validation_data=(val_img_patches, val_mask_patches),
        callbacks=[checkpoint, early],
    )

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


## 📊 Resultados das Métricas de Avaliação

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
- Ela erra menos (menor loss),
- Classifica com maior precisão geral (acurácia),
- Tem maior sensibilidade (detecta melhor os positivos),
- Maior especificidade (detecta melhor os negativos),
- Melhor separação entre classes (AUC-ROC).
  
Portanto, U-Net++ é a melhor escolha com base nesses resultados.

### Comparaçõa das Imagens de ambos os modelos:
<img width="1710" height="464" alt="image" src="https://github.com/user-attachments/assets/e836580d-1b74-40cc-a9db-3acde81a2d17" />
---

## 📌 Observação

O `threshold=0.1` foi utilizado para binarizar as predições. Isso ajuda a captar vasos mais finos, mas deve ser ajustado conforme necessário. Um valor muito baixo pode aumentar o número de falsos positivos.

Para visualização as imagens deve ter o tamanho próximo ao originais, apenas para o treinamento, foi feito com proporções 128x128.
Ao utilizar as proporções 128x128 sem a extração de patchs, o resultado fica assim:

<img width="873" height="313" alt="image" src="https://github.com/user-attachments/assets/74c5e370-b546-4893-b69a-9a66104d4352" />

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
