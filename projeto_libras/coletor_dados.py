"""
Coletor de Dados para Libras - Versão com MediaPipe Tasks API
Agora com suporte a sequencias de frames para capturar movimento
"""
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ColetorLibras:
    def __init__(self):
        self.letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.letra_atual = 'A'
        
        # Configuracoes para captura de sequencias
        self.sequencia_tamanho = 20  # 20 frames por sequencia
        self.sequencia_atual = deque(maxlen=self.sequencia_tamanho)
        self.gravando = False
        self.contador_sequencias = {letra: 0 for letra in self.letras}
        
        # Inicializa webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            print("ERRO: Webcam nao encontrada!")
            exit()
        
        # Configura o detector de maos usando Tasks API
        try:
            base_options = python.BaseOptions(model_asset_path="modelos/hand_landmarker.task")
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=1,
                result_callback=self.resultado_mao,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.detector = vision.HandLandmarker.create_from_options(options)
            print("Detector de maos inicializado com sucesso!")
        except Exception as e:
            print(f"Erro ao inicializar detector: {e}")
            print("Tentando metodo alternativo...")
            self.detector = None
        
        # Dados
        self.dados = []
        self.colunas = self._gerar_colunas()
        
        # Controle (mantido para compatibilidade)
        self.ultima_coleta = 0
        self.pausa = 0.3
        
    def _gerar_colunas(self):
        """Gera nomes das colunas para o dataset sequencial"""
        colunas = ['letra']
        for frame_idx in range(self.sequencia_tamanho):
            for ponto_idx in range(21):
                colunas.extend([
                    f'f{frame_idx}_x{ponto_idx}',
                    f'f{frame_idx}_y{ponto_idx}',
                    f'f{frame_idx}_z{ponto_idx}'
                ])
        return colunas
    
    def resultado_mao(self, result, output_image, timestamp_ms):
        """Callback acionado quando o MediaPipe processa um frame"""
        if result.hand_landmarks and self.gravando:
            hand = result.hand_landmarks[0]
            landmarks = []
            for lm in hand:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            self.sequencia_atual.append(landmarks)
    
    def salvar_sequencia(self):
        """Salva a sequencia completa no dataset"""
        if len(self.sequencia_atual) < self.sequencia_tamanho:
            print(f"Sequencia incompleta: {len(self.sequencia_atual)}/{self.sequencia_tamanho}")
            return
        
        sequencia_array = np.array(self.sequencia_atual)
        
        amostra = [self.letra_atual]
        for frame in sequencia_array:
            amostra.extend(frame.flatten())
        
        self.dados.append(amostra)
        self.contador_sequencias[self.letra_atual] += 1
        print(f"{self.letra_atual}: Sequencia {self.contador_sequencias[self.letra_atual]} salva!")
        
        self.sequencia_atual.clear()
        self.gravando = False
    
    def _extrair_landmarks(self, hand_landmarks):
        """Extrai landmarks do formato da Tasks API (mantido para compatibilidade)"""
        landmarks = []
        for lm in hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    
    def salvar_dados(self):
        """Salva todas as sequencias em CSV"""
        if not self.dados:
            print("Nenhum dado para salvar!")
            return
        
        os.makedirs('dados', exist_ok=True)
        df = pd.DataFrame(self.dados, columns=self.colunas)
        
        arquivo = 'dados/dataset_libras.csv'
        if os.path.exists(arquivo):
            df_existente = pd.read_csv(arquivo)
            df = pd.concat([df_existente, df], ignore_index=True)
        
        df.to_csv(arquivo, index=False)
        print(f"\nDados salvos! Total: {len(df)} amostras (sequencias)")
        
        print("\nAmostras por letra:")
        for letra in sorted(df['letra'].unique()):
            count = len(df[df['letra'] == letra])
            print(f"  Letra {letra}: {count}")
    
    def detectar_maos(self, frame):
        """Detecta maos usando Tasks API (mantido para compatibilidade)"""
        if self.detector is None:
            return None, False
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            return detection_result.hand_landmarks[0], True
        return None, False
    
    def desenhar_landmarks(self, frame, hand_landmarks):
        """Desenha os landmarks manualmente"""
        h, w = frame.shape[:2]
        
        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        connections = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (0,9), (9,10), (10,11), (11,12),
            (0,13), (13,14), (14,15), (15,16),
            (0,17), (17,18), (18,19), (19,20),
            (5,9), (9,13), (13,17)
        ]
        
        for connection in connections:
            idx1, idx2 = connection
            if idx1 < len(hand_landmarks) and idx2 < len(hand_landmarks):
                x1 = int(hand_landmarks[idx1].x * w)
                y1 = int(hand_landmarks[idx1].y * h)
                x2 = int(hand_landmarks[idx2].x * w)
                y2 = int(hand_landmarks[idx2].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    def executar(self):
        print("=" * 50)
        print("COLETOR DE DADOS - LIBRAS (COM SEQUENCIAS)")
        print("=" * 50)
        print(f"Tamanho da sequencia: {self.sequencia_tamanho} frames")
        print("\nINSTRUCOES:")
        print("  • ESPACO: Iniciar/Parar gravacao da sequencia")
        print("  • N: Proxima letra")
        print("  • P: Letra anterior")
        print("  • S: Salvar e sair")
        print("  • ESC: Sair sem salvar")
        print("=" * 50)
        
        letras_movimento = ['H', 'J', 'X', 'Z']
        print(f"\nDICA: Letras com movimento: {', '.join(letras_movimento)}")
        print("Para estas, faca o movimento completo durante a gravacao!\n")
        
        if self.detector is None:
            print("\nATENCAO: Detector nao inicializado!")
            print("Pressione qualquer tecla para continuar mesmo assim...")
            cv2.waitKey(0)
        
        # Pega as dimensões do frame para passar ao MediaPipe (corrige o warning)
        ret, frame_teste = self.cap.read()
        if ret:
            h, w = frame_teste.shape[:2]
            print(f"Dimensoes da imagem: {w}x{h}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            if self.detector is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # CORREÇÃO: Criar mp.Image com as dimensões corretas
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                # Envia para detecção com timestamp
                self.detector.detect_async(mp_image, int(time.time() * 1000))
            
            h, w = frame.shape[:2]
            
            if self.gravando:
                status = "GRAVANDO"
                cor_status = (0, 0, 255)
            else:
                status = "PRONTO"
                cor_status = (0, 255, 0)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (400, 220), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, f"Letra: {self.letra_atual}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Status: {status}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_status, 2)
            cv2.putText(frame, f"Frames: {len(self.sequencia_atual)}/{self.sequencia_tamanho}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Total {self.letra_atual}: {self.contador_sequencias[self.letra_atual]}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
            
            if self.gravando:
                progresso = len(self.sequencia_atual) / self.sequencia_tamanho
                barra_x, barra_y, barra_w, barra_h = 10, 175, 300, 15
                cv2.rectangle(frame, (barra_x, barra_y), 
                            (barra_x + barra_w, barra_y + barra_h), (100, 100, 100), -1)
                cv2.rectangle(frame, (barra_x, barra_y), 
                            (barra_x + int(barra_w * progresso), barra_y + barra_h), (0, 255, 0), -1)
            
            total_geral = sum(self.contador_sequencias.values())
            meta = len(self.letras) * 30
            percentual = (total_geral / meta) * 100 if meta > 0 else 0
            
            cv2.putText(frame, f"Total geral: {total_geral}/{meta} ({percentual:.1f}%)", 
                       (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            instrucoes = [
                "ESPACO: gravar/parar",
                "N: prox | P: ant",
                "S: salvar | ESC: sair"
            ]
            y_inst = frame.shape[0] - 60
            for i, texto in enumerate(instrucoes):
                cv2.putText(frame, texto, (10, y_inst + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            idx_atual = self.letras.index(self.letra_atual)
            inicio = max(0, idx_atual - 4)
            fim = min(len(self.letras), idx_atual + 5)
            
            y_mapa = frame.shape[0] - 80
            x_mapa = frame.shape[1] - 300
            for i, letra in enumerate(self.letras[inicio:fim]):
                cor = (255, 255, 0) if letra == self.letra_atual else (200, 200, 200)
                cv2.putText(frame, letra, (x_mapa + i*35, y_mapa),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
                if self.contador_sequencias[letra] > 0:
                    cv2.putText(frame, str(self.contador_sequencias[letra]), 
                               (x_mapa + i*35, y_mapa + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            cv2.imshow('Coletor Libras', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:
                if not self.gravando:
                    self.sequencia_atual.clear()
                    self.gravando = True
                    print(f"\nGravando letra {self.letra_atual}...")
                else:
                    self.salvar_sequencia()
            
            elif key == ord('n') or key == ord('N'):
                idx = self.letras.index(self.letra_atual)
                self.letra_atual = self.letras[(idx + 1) % len(self.letras)]
                print(f"Mudou para letra: {self.letra_atual}")
            
            elif key == ord('p') or key == ord('P'):
                idx = self.letras.index(self.letra_atual)
                self.letra_atual = self.letras[(idx - 1) % len(self.letras)]
                print(f"Mudou para letra: {self.letra_atual}")
            
            elif key == ord('s') or key == ord('S'):
                if self.gravando:
                    self.salvar_sequencia()
                self.salvar_dados()
                break
            
            elif key == 27:
                print("Saindo sem salvar...")
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        coletor = ColetorLibras()
        coletor.executar()
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
        input("Pressione Enter para sair...") 