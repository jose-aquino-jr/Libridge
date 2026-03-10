# 🤟 Libridge
### Sistema de Reconhecimento de Libras com Inteligência Artificial

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-HandTracking-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Libridge** é um sistema de **reconhecimento de Libras (Língua Brasileira de Sinais)** utilizando **Visão Computacional e Deep Learning**.

O projeto utiliza:

- **MediaPipe Hand Tracking** para detectar mãos
- **TensorFlow / Keras (LSTM)** para reconhecimento de sequências
- **OpenCV** para captura de vídeo
- **Machine Learning sequencial** para identificar movimentos e sinais

O objetivo é criar uma **ponte entre comunicação em Libras e sistemas computacionais.**

---

# 📋 Requisitos

Para executar o projeto corretamente, recomenda-se:

- **Python 3.11**
- Webcam
- Sistema operacional: **Windows, Linux ou Mac**

⚠️ Algumas bibliotecas como **TensorFlow** e **MediaPipe** podem apresentar incompatibilidade com versões mais novas do Python.  
Por isso o projeto foi desenvolvido e testado utilizando **Python 3.11**.

Para verificar sua versão do Python:

```bash
python --version
```

---

# 📸 Demonstração

## Detecção de mãos

![Detecção](projeto_libras/Images/hand_detection.png)

---

## Interface do coletor de dados

![Coletor](projeto_libras/Images/data_collector.png)

---

## Treinamento da IA

![Treinamento](projeto_libras/Images/training_accuracy.png)

---

## Matriz de Confusão

![Confusão](projeto_libras/Images/confusion_matrix.png)

---

# 🧠 Como funciona

O Libridge funciona em **3 etapas principais**:

### 1️⃣ Coleta de dados

A webcam captura **sequências de movimento da mão**.

Cada sequência contém:

20 frames  
21 landmarks da mão  
3 coordenadas (x,y,z)

Total por frame:

21 × 3 = 63 features

Total por sequência:

20 × 63 = 1260 features

Esses dados são salvos no dataset:

dados/dataset_libras.csv

---

### 2️⃣ Treinamento da IA

A rede neural usa **LSTM (Long Short-Term Memory)** para aprender **movimentos ao longo do tempo**.

Arquitetura:

LSTM (128)  
Dropout  

LSTM (64)  
Dropout  

LSTM (32)  
Dropout  

Dense (64)  
Dense (32)  

Softmax

Saída:

26 classes (A-Z)

---

### 3️⃣ Detecção em tempo real

O sistema:

1. Captura frames da webcam
2. Detecta a mão
3. Extrai landmarks
4. Forma uma sequência
5. A IA prevê a letra

Tudo em **tempo real**.

---

# 🗂 Estrutura do Projeto

```
libridge/

├── dados/
│ └── dataset_libras.csv
│
├── modelos/
│ ├── melhor_modelo_libras.keras
│ ├── label_encoder.pkl
│ └── config.pkl
│
├── resultados/
│ ├── historico_treinamento.png
│ └── matriz_confusao.png
│
├── coletor_dados.py
├── treinador.py
├── detector_tempo_real.py
├── main.py
│
└── README.md
```

---

# ⚙️ Instalação

## 1️⃣ Clonar o projeto

```bash
git clone https://github.com/seu-usuario/libridge.git
cd libridge
```

---

## 2️⃣ Criar ambiente virtual (Python 3.11 recomendado)

### Linux / Mac

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Windows

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

---

## 3️⃣ Instalar dependências

Recomendado usar o comando abaixo para garantir que o **pip correto do ambiente virtual** seja utilizado:

```bash
python -m pip install -r requirements.txt
```

Caso o arquivo **requirements.txt** não exista, instale manualmente:

```bash
pip install tensorflow opencv-python mediapipe pandas numpy scikit-learn seaborn matplotlib joblib scipy
```

---

# ▶️ Como usar

Execute o menu principal:

```bash
python main.py
```

Menu:

1. Coletar dados  
2. Treinar IA  
3. Detectar em tempo real  
4. Estatísticas  
5. Sair  

---

# 📊 1️⃣ Coletar dados

Captura sequências de movimentos para treinar a IA.

```bash
python coletor_dados.py
```

⚠️ **Aviso importante**

O repositório **já inclui um dataset base** em:

```
dados/dataset_libras.csv
```

Portanto **não é obrigatório coletar os dados novamente** para treinar o modelo.

Você pode:

- utilizar o dataset existente
- complementar com novos dados
- ou criar seu próprio dataset

Controles:

ESPACO → iniciar/parar gravação  
N → próxima letra  
P → letra anterior  
S → salvar dataset  
ESC → sair  

Recomendado:

30 sequências por letra

Total ideal:

26 letras × 30 = 780 amostras

---

# 🧠 2️⃣ Treinar modelo

Treina a rede neural LSTM.

```bash
python treinador.py
```

Saídas:

modelos/melhor_modelo_libras.keras  
modelos/label_encoder.pkl  
modelos/config.pkl  

Resultados gerados:

resultados/historico_treinamento.png  
resultados/matriz_confusao.png  

---

# 🎥 3️⃣ Detecção em tempo real

Executa reconhecimento de Libras via webcam.

```bash
python detector_tempo_real.py
```

Interface mostra:

✔ letra detectada  
✔ confiança  
✔ FPS  
✔ landmarks da mão  

---

# 📊 Estatísticas do dataset

Mostra:

- total de amostras
- letras existentes
- sequências por letra

---

# 🧪 Exemplo de dataset

```
letra,f0_x0,f0_y0,f0_z0,f0_x1,f0_y1,f0_z1...
A,0.89,0.73,-0.0000009,0.79,0.73,-0.02...
A,0.88,0.72,-0.0000009,0.78,0.73,-0.03...
```

Cada linha representa uma sequência completa de movimento.

---

# 📈 Resultados

Exemplo obtido no treinamento:

Accuracy: ~99%  
Loss: baixo  

Matriz de confusão quase perfeita.

---

# 🚀 Tecnologias usadas

Python  

TensorFlow / Keras  

OpenCV  

MediaPipe  

NumPy  

Pandas  

Scikit-Learn  

Matplotlib  

Seaborn  

---

# 🧩 Melhorias futuras

Reconhecimento de palavras  

Tradução Libras → Texto  

Tradução Libras → Voz  

Modelo mais robusto (Transformer)  

Dataset maior  

Interface web  

---

# 🤝 Contribuindo

Pull requests são bem-vindos.

1️⃣ Fork o projeto  

2️⃣ Crie uma branch

```bash
git checkout -b minha-feature
```

3️⃣ Commit

```bash
git commit -m "Nova feature"
```

4️⃣ Push

```bash
git push origin minha-feature
```

5️⃣ Abra um Pull Request

---

# 📜 Licença

Este projeto está sob a licença MIT.

---

# 👨‍💻 Autor

Projeto desenvolvido por:

José Aquino Junior

Estudante de tecnologia Desenvolvedor de Software e entusiasta de IAs.

---

# ⭐ Apoie o projeto

Se gostou do projeto:

⭐ Deixe uma estrela no repositório  
🤝 Compartilhe  
💡 Contribua
