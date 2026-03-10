"""
Detector em Tempo Real para Libras - Versão Rápida
"""
import cv2
import numpy as np
import joblib
from collections import deque
import time
import mediapipe as mp
from scipy import stats
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class DetectorLibras:
    def __init__(self):
        self.carregar_modelos()
        self.inicializar_camera()
        self.inicializar_detector()
        
        self.buffer_frames = deque(maxlen=20)
        self.ultima_letra = "?"
        self.confianca = 0
        self.mao_detectada = False
        self.ultimos_landmarks = None
        
        self.fps = 0
        self.fps_count = 0
        self.fps_tempo = time.time()
        
        self.letra_mostrada = "?"
        self.tempo_mostra = 0
        self.ultima_predicao = 0
        self.intervalo_predicao = 0.05  # 50ms entre predicoes

    def carregar_modelos(self):
        try:
            self.modelo = tf.keras.models.load_model('modelos/melhor_modelo_libras.keras')
            self.label_encoder = joblib.load('modelos/label_encoder.pkl')
            self.sequencia_tamanho = 20
            print("Modelo carregado")
        except Exception as e:
            print(f"Erro modelo: {e}")
            exit()

    def inicializar_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            print("Camera nao encontrada")
            exit()

    def inicializar_detector(self):
        try:
            base_options = python.BaseOptions(model_asset_path="modelos/hand_landmarker.task")
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.3
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            print("Detector ok")
        except Exception as e:
            print(f"Erro detector: {e}")
            exit()

    def processar_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            if not self.mao_detectada:
                self.buffer_frames.clear()
            
            self.mao_detectada = True
            self.ultimos_landmarks = detection_result.hand_landmarks[0]
            
            landmarks = []
            for lm in self.ultimos_landmarks:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            self.buffer_frames.append(landmarks)
            
            # Faz predicao a CADA frame, mesmo com buffer incompleto
            if len(self.buffer_frames) >= 5:  # Comeca a predizer com apenas 5 frames
                self.fazer_predicao_rapida()
        else:
            self.mao_detectada = False
            self.ultimos_landmarks = None
            self.buffer_frames.clear()
            self.ultima_letra = "?"
            self.letra_mostrada = "?"

    def fazer_predicao_rapida(self):
        # Usa os frames disponiveis (repetindo se necessario)
        frames_disponiveis = list(self.buffer_frames)
        
        if len(frames_disponiveis) < self.sequencia_tamanho:
            # Repete o ultimo frame para completar a sequencia
            ultimo_frame = frames_disponiveis[-1]
            frames_faltando = self.sequencia_tamanho - len(frames_disponiveis)
            frames_completos = frames_disponiveis + [ultimo_frame] * frames_faltando
        else:
            frames_completos = frames_disponiveis[-self.sequencia_tamanho:]
        
        sequencia = np.array(frames_completos)
        sequencia = sequencia.reshape(1, self.sequencia_tamanho, -1)
        
        pred_proba = self.modelo.predict(sequencia, verbose=0)[0]
        pred_classe = np.argmax(pred_proba)
        confianca = np.max(pred_proba)
        
        if confianca > 0.6:
            letra = self.label_encoder.inverse_transform([pred_classe])[0]
            
            # Atualiza imediatamente
            self.ultima_letra = letra
            self.confianca = confianca
            self.letra_mostrada = letra
            self.tempo_mostra = time.time()

    def desenhar_landmarks(self, frame):
        if self.ultimos_landmarks is None:
            return
        
        h, w = frame.shape[:2]
        
        for lm in self.ultimos_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0, 200, 0), -1)
        
        connections = [
            (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
            (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
            (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
        ]
        
        for idx1, idx2 in connections:
            if idx1 < len(self.ultimos_landmarks) and idx2 < len(self.ultimos_landmarks):
                x1 = int(self.ultimos_landmarks[idx1].x * w)
                y1 = int(self.ultimos_landmarks[idx1].y * h)
                x2 = int(self.ultimos_landmarks[idx2].x * w)
                y2 = int(self.ultimos_landmarks[idx2].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 100, 0), 1)

    def desenhar_interface(self, frame):
        h, w = frame.shape[:2]
        
        cv2.putText(frame, f"FPS: {int(self.fps)}", (w-80, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        status = "OK" if self.mao_detectada else "SEM MAO"
        cor = (0, 255, 0) if self.mao_detectada else (0, 0, 255)
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
        
        # Mostra buffer
        if self.mao_detectada:
            cv2.putText(frame, f"B:{len(self.buffer_frames)}", (w-80, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Letra grande
        if self.letra_mostrada != "?" and time.time() - self.tempo_mostra < 1.5:
            letra = self.letra_mostrada
            tamanho = cv2.getTextSize(letra, cv2.FONT_HERSHEY_SIMPLEX, 5, 5)[0]
            x = (w - tamanho[0]) // 2
            y = (h + tamanho[1]) // 2 - 50
            
            cv2.putText(frame, letra, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)
            
            # Barra de confianca
            barra_x, barra_y = x, y + 40
            cv2.rectangle(frame, (barra_x, barra_y), (barra_x + 150, barra_y + 10), (50, 50, 50), -1)
            cv2.rectangle(frame, (barra_x, barra_y), 
                         (barra_x + int(150 * self.confianca), barra_y + 10), (0, 255, 0), -1)
            cv2.putText(frame, f"{int(self.confianca*100)}%", (barra_x + 160, barra_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def executar(self):
        print("DETECTOR LIBRAS - MODO RAPIDO")
        print("ESC para sair")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            
            self.processar_frame(frame)
            
            self.fps_count += 1
            if time.time() - self.fps_tempo >= 1.0:
                self.fps = self.fps_count
                self.fps_count = 0
                self.fps_tempo = time.time()
            
            self.desenhar_landmarks(frame)
            self.desenhar_interface(frame)
            
            cv2.imshow('Detector', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DetectorLibras()
    detector.executar()