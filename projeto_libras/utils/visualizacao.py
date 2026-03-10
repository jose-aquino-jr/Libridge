"""
Módulo com funções auxiliares para visualização
"""
import cv2
import numpy as np
import mediapipe as mp

def desenhar_info_mao(frame, hand_landmarks, largura, altura):
    """
    Desenha informações detalhadas da mão
    """
    # Pontos dos dedos
    pontas_dedos = [4, 8, 12, 16, 20]  # Polegar, indicador, médio, anelar, mínimo
    
    for i, ponto in enumerate(pontas_dedos):
        if ponto < len(hand_landmarks.landmark):
            lm = hand_landmarks.landmark[ponto]
            x, y = int(lm.x * largura), int(lm.y * altura)
            
            # Desenha círculo na ponta do dedo
            cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)
            
            # Nome do dedo
            nomes = ['Polegar', 'Indicador', 'Medio', 'Anelar', 'Minimo']
            cv2.putText(frame, nomes[i], (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def criar_grade_mao(landmarks, largura, altura):
    """
    Cria uma grade 3D da mão para visualização
    """
    grade = np.zeros((21, 3))
    for i, lm in enumerate(landmarks.landmark):
        grade[i] = [lm.x, lm.y, lm.z]
    return grade

def desenhar_mao_3d(frame, grade, offset_x=400, offset_y=200, escala=100):
    """
    Desenha uma representação 3D simplificada da mão
    """
    # Conexões da mão (MediaPipe)
    conexoes = mp.solutions.hands.HAND_CONNECTIONS
    
    # Desenha conexões
    for conexao in conexoes:
        pt1 = grade[conexao[0]]
        pt2 = grade[conexao[1]]
        
        x1 = int(offset_x + pt1[0] * escala)
        y1 = int(offset_y + pt1[1] * escala)
        x2 = int(offset_x + pt2[0] * escala)
        y2 = int(offset_y + pt2[1] * escala)
        
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Desenha pontos
    for i, pt in enumerate(grade):
        x = int(offset_x + pt[0] * escala)
        y = int(offset_y + pt[1] * escala)
        cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        cv2.putText(frame, str(i), (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def desenhar_histograma_confianca(frame, confiancas, classes):
    """
    Desenha histograma das confianças para cada classe
    """
    altura, largura = frame.shape[:2]
    hist_x = largura - 250
    hist_y = 50
    hist_w = 200
    hist_h = 150
    
    # Fundo
    cv2.rectangle(frame, (hist_x, hist_y), 
                 (hist_x + hist_w, hist_y + hist_h), 
                 (50, 50, 50), -1)
    
    # Título
    cv2.putText(frame, "Confiancas por Classe", (hist_x + 10, hist_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Barras
    bar_w = hist_w // len(classes) - 2
    for i, (classe, conf) in enumerate(zip(classes, confiancas)):
        x = hist_x + i * (bar_w + 2) + 2
        bar_h = int(conf * (hist_h - 40))
        y = hist_y + hist_h - 20 - bar_h
        
        # Cor baseada na confiança
        cor = (0, int(255 * conf), 0)
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), cor, -1)
        
        # Label da classe
        cv2.putText(frame, classe, (x, hist_y + hist_h - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def criar_overlay_info(frame, texto, posicao='top-right'):
    """
    Cria overlay com informações na tela
    """
    h, w = frame.shape[:2]
    
    if posicao == 'top-right':
        x, y = w - 200, 10
    elif posicao == 'top-left':
        x, y = 10, 10
    elif posicao == 'bottom-right':
        x, y = w - 200, h - 100
    else:  # bottom-left
        x, y = 10, h - 100
    
    # Fundo semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + 190, y + 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Texto
    y_offset = y + 20
    for linha in texto.split('\n'):
        cv2.putText(frame, linha, (x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
    
    return frame