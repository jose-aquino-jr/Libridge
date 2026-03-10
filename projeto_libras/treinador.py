"""
Treinador com LSTM para reconhecimento de sequencias de Libras
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class TreinadorLibras:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        
        # Cria pastas necessarias
        if not os.path.exists('modelos'):
            os.makedirs('modelos')
        if not os.path.exists('resultados'):
            os.makedirs('resultados')
    
    def carregar_dados(self, arquivo='dados/dataset_libras.csv'):
        """Carrega e prepara os dados sequenciais"""
        print("\nCarregando dados...")
        
        if not os.path.exists(arquivo):
            print(f"Arquivo {arquivo} nao encontrado!")
            return False
        
        df = pd.read_csv(arquivo)
        print(f"Dados carregados: {len(df)} sequencias")
        print(f"Letras encontradas: {sorted(df['letra'].unique())}")
        
        # Determina o tamanho da sequencia baseado nas colunas
        colunas = df.columns
        num_colunas_landmarks = len([c for c in colunas if 'x' in c or 'y' in c or 'z' in c])
        self.sequencia_tamanho = num_colunas_landmarks // 63  # 21 landmarks * 3 coordenadas
        self.num_landmarks = 63  # 21 * 3
        
        print(f"Sequencias de {self.sequencia_tamanho} frames detectadas")
        
        # Prepara X e y
        X = []
        y = []
        
        for _, row in df.iterrows():
            letra = row['letra']
            valores = row.drop('letra').values.astype(np.float32)
            
            # Remodela para (sequencia_tamanho, num_landmarks)
            try:
                sequencia = valores.reshape(self.sequencia_tamanho, self.num_landmarks)
                X.append(sequencia)
                y.append(letra)
            except:
                print(f"Erro ao processar linha, pulando...")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nFormato dos dados:")
        print(f"  X shape: {X.shape} (amostras, frames, landmarks)")
        print(f"  y shape: {y.shape}")
        
        # Codifica labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Divide em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\nDivisao dos dados:")
        print(f"  Treino: {len(self.X_train)} sequencias")
        print(f"  Teste: {len(self.X_test)} sequencias")
        
        return True
    
    def criar_modelo(self):
        """Cria modelo LSTM para classificacao de sequencias"""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(self.sequencia_tamanho, self.num_landmarks)),
            layers.Dropout(0.3),
            
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            
            layers.LSTM(32),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            
            layers.Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def treinar(self, epochs=150, batch_size=32):
        """Treina o modelo LSTM"""
        print("\nCriando modelo LSTM...")
        self.model = self.criar_modelo()
        self.model.summary()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'modelos/melhor_modelo_libras.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("\nIniciando treinamento...")
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def avaliar(self, history):
        """Avalia o modelo treinado"""
        print("\nAvaliando modelo...")
        
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"\nAcuracia no teste: {test_acc:.4f}")
        print(f"Loss no teste: {test_loss:.4f}")
        
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\nRelatorio por classe:")
        print(classification_report(
            self.y_test, 
            y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        self.plotar_historico(history)
        self.plotar_matriz_confusao(self.y_test, y_pred)
    
    def plotar_historico(self, history):
        """Plota o historico de treinamento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history.history['accuracy'], label='Treino')
        ax1.plot(history.history['val_accuracy'], label='Validacao')
        ax1.set_title('Acuracia durante treinamento')
        ax1.set_xlabel('Epocas')
        ax1.set_ylabel('Acuracia')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(history.history['loss'], label='Treino')
        ax2.plot(history.history['val_loss'], label='Validacao')
        ax2.set_title('Loss durante treinamento')
        ax2.set_xlabel('Epocas')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'resultados/historico_treinamento_{timestamp}.png')
        plt.show()
    
    def plotar_matriz_confusao(self, y_true, y_pred):
        """Plota matriz de confusao"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        
        plt.title('Matriz de Confusao - LSTM')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'resultados/matriz_confusao_{timestamp}.png')
        plt.show()
    
    def salvar_modelo(self):
        """Salva o modelo e os encoders"""
        self.model.save('modelos/melhor_modelo_libras.keras')
        
        joblib.dump(self.label_encoder, 'modelos/label_encoder.pkl')
        
        config = {
            'sequencia_tamanho': self.sequencia_tamanho,
            'num_landmarks': self.num_landmarks,
            'classes': list(self.label_encoder.classes_)
        }
        joblib.dump(config, 'modelos/config.pkl')
        
        print("\nModelo salvo em 'modelos/'")
    
    def executar(self):
        """Executa todo o pipeline"""
        print("=" * 50)
        print("TREINADOR DE IA - LIBRAS (LSTM)")
        print("=" * 50)
        
        if not self.carregar_dados():
            return
        
        history = self.treinar(epochs=150, batch_size=32)
        
        self.avaliar(history)
        
        self.salvar_modelo()
        
        print("\nTreinamento concluido com sucesso!")

if __name__ == "__main__":
    treinador = TreinadorLibras()
    treinador.executar()