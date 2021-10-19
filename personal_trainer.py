################################################################################################
#                                                                                              #
#  Toda a lógica mais detalhada está presente no arquivo "Contador de Dedos.ipynb"             #
#                                                                                              #
#  Em caso de dúvidas, consultar a documentação:                                               #
#      - "Aula 3 - Estimativa de pose (Introdução).ipynb" no link abaixo.                      #
#                                                                                              #
#  GitHub: https://github.com/GTL98/curso-completo-de-visao-computacional-avancada-com-python  #
#                                                                                              #
################################################################################################


# Importar as bibliotecas
import cv2
import numpy as np
import time
import estimativa_pose as ep


# Definir o tamanho da tela
largura_tela = 640
altura_tela = 480


# Taxa de frame (FPS)
tempo_atual = 0
tempo_anterior = 0


# Módulo DetectorPose
detector = ep.DetectorPose(deteccao_confianca=0.75, rastreamento_confianca=0.75)


# Repetições e movimento completo
repeticoes = 0
completo = 0


# Captura de vídeo
cap = cv2.VideoCapture(0)
cap.set(3, largura_tela)
cap.set(4, altura_tela)
lado = 0

while True:
    sucesso, imagem = cap.read()
    imagem = cv2.resize(imagem, (largura_tela, altura_tela))
    imagem = detector.encontrar_pose(imagem, False)
    lista_landmark = detector.encontrar_posicao(imagem, False)
    
    # Pegar as landmarks que usaremos
    if lista_landmark:
        if lado == 0:
            # Braço esquerdo
            angulo = detector.encontrar_angulo(imagem, 11, 13, 15)
            porcentagem = np.interp(angulo, [200, 310], [0, 100])
            barra = np.interp(angulo, [205, 305], [370, 120])
            
            # Checar as repetições
            cor = (255, 0, 255)
            if porcentagem == 100:
                cor = (0, 255, 0)
                if completo == 1:
                    repeticoes += 0.5
                    completo = 0
            if porcentagem == 0:
                cor = (0, 255, 0)
                if completo == 0:
                    repeticoes += 0.5
                    completo = 1
                
            # Desenhar o retângulo da porcentagem do movimento
            cv2.rectangle(imagem, (30, 120), (65, 370), cor, 3)
            
            # Desenhar o preenchimento da porcentagem do movimento
            cv2.rectangle(imagem, (30, int(barra)), (65, 370), cor, cv2.FILLED)
            
            # Escrever a porcentagem da barra
            cv2.putText(imagem, f'{int(porcentagem)}%', (40, 100), cv2.FONT_HERSHEY_PLAIN, 3, cor, 3)
                
            # Colocar o retângulo onde será mostrado o número de repetições
            cv2.rectangle(imagem, (480, 400), (640, 480), (0, 0, 0), cv2.FILLED)
            
            # Colocar o número de repetições no retângulo
            cv2.putText(imagem, f'{int(repeticoes)}', (490 ,470), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 10)
        
        elif lado == 1:
            # Braço direito
            angulo = detector.encontrar_angulo(imagem, 12, 14, 16)
            porcentagem = np.interp(angulo, [50, 170], [100, 0])
            barra = np.interp(angulo, [50, 170], [120, 370])
            
            # Checar as repetições
            cor = (255, 0, 255)
            if porcentagem == 100:
                cor = (0, 255, 0)
                if completo == 1:
                    repeticoes += 0.5
                    completo = 0
            if porcentagem == 0:
                cor = (0, 255, 0)
                if completo == 0:
                    repeticoes += 0.5
                    completo = 1
                
            # Desenhar o retângulo da porcentagem do movimento
            cv2.rectangle(imagem, (30, 120), (65, 370), cor, 3)
            
            # Desenhar o preenchimento da porcentagem do movimento
            cv2.rectangle(imagem, (30, int(barra)), (65, 370), cor, cv2.FILLED)
            
            # Escrever a porcentagem da barra
            cv2.putText(imagem, f'{int(porcentagem)}%', (40, 100), cv2.FONT_HERSHEY_PLAIN, 3, cor, 3)
                
            # Colocar o retângulo onde será mostrado o número de repetições
            cv2.rectangle(imagem, (480, 400), (640, 480), (0, 0, 0), cv2.FILLED)
            
            # Colocar o número de repetições no retângulo
            cv2.putText(imagem, f'{int(repeticoes)}', (490 ,470), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 10)
            
    # Configurar o FPS
    tempo_atual = time.time()
    fps = 1/(tempo_atual - tempo_anterior)
    tempo_anterior = tempo_atual
    
    # Mostrar o FPS na tela
    cv2.putText(imagem, f'FPS: {int(fps)}', (425, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    # Mostrar a imagem na tela
    cv2.imshow('Imagem', imagem)
    
    # Terminar o loop
    if cv2.waitKey(1) & 0xFF == ord('s'):
        repeticoes = 0
        completo = 0
        break
        
# Fechar a tela de captura
cap.release()
cv2.destroyAllWindows()