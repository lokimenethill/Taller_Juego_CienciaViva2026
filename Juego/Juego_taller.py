from this import s

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import random

# Variables
puntuacion = 0
numeroDeVidas = 3
maxFrutas = 30
maxBombas = 10
listaFrutas = []
listaBombas = []
tamFruta = 30
incrementoFruta = 1
disminucionFruta = 2
tamBomba = 30
tiempoLimite = 10
camara = 1
margen = 100



def calcularCentro(ymin, ymax, xmin, xmax):
    return (int((xmin + xmax) / 2), int((ymin + ymax) / 2))


def calcularTam(ymin, ymax, xmin, xmax):
    altura = ymax - ymin
    base = xmax - xmin
    return altura, base


def tamRectangulo(base, altura):
    return base * altura

def calcularRaizCuadrada(num):
    return math.sqrt(num)
# ====================== Qué falta aquí?  =====================
def calcularDistancia2Puntos(x1, y1, x2, y2):
    return calcularRaizCuadrada(2)
# ====================== Qué falta aquí?  =====================
def haySolapamiento(nR, nL, radio):
    for bomba in listaBombas:
        ## ------>
        if distancia < (radio + tamBomba):
            return True

    for fruta in listaFrutas:
        ## ------>
        if distancia < (radio + tamFruta):
            return True

    return False


def generarPosicionValida(radio, x, y, max_intentos=1000):
    for i in range(max_intentos):
        nR = random.randint(margen, x - margen)
        nL = random.randint(margen, y - margen)
        if not haySolapamiento(nR, nL, radio):
            return nR, nL
    return nR, nL


def generarBombas(frame, x, y, c):
    while len(listaBombas) < maxBombas:
        pos = generarPosicionValida(tamBomba, x, y)
        listaBombas.append(pos)
    for bomba in listaBombas:
        cv2.circle(frame, (bomba[0], bomba[1]), tamBomba, (0, 0, 255), cv2.FILLED)


def generarFrutas(frame, x, y, c):
    while len(listaFrutas) < maxFrutas:
        pos = generarPosicionValida(tamFruta, x, y)
        listaFrutas.append(pos)
    for fruta in listaFrutas:
        cv2.circle(frame, (fruta[0], fruta[1]), tamFruta, (255, 0, 0), cv2.FILLED)





class HandTrackingDynamic:
    def __init__(self, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # Ideal para video o webcam
            num_hands=self.__maxHands__,
            min_hand_detection_confidence=self.__detectionCon__,
            min_hand_presence_confidence=self.__detectionCon__,
            min_tracking_confidence=self.__trackCon__
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # El modo VIDEO requiere que enviemos timestamps crecientes
        self.frame_count = 0

        self.results = None
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmsList = []

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convertimos la imagen de OpenCV al formato de MediaPipe Tasks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

        # Simulamos un timestamp creciente (aproximadamente 30 FPS)
        self.frame_count += 1
        timestamp_ms = self.frame_count * 33

        # Procesamos el frame
        self.results = self.detector.detect_for_video(mp_image, timestamp_ms)

        # Si detecta manos y queremos dibujar
        if self.results and self.results.hand_landmarks:
            if draw:
                # Al no usar drawing_utils de la versión anterior,
                # dibujamos las conexiones manualmente para evitar incompatibilidades.
                conexiones = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17)
                ]
                h, w, _ = frame.shape
                for mano_detectada in self.results.hand_landmarks:
                    for conexion in conexiones:
                        pt1 = mano_detectada[conexion[0]]
                        pt2 = mano_detectada[conexion[1]]

                        x1, y1 = int(pt1.x * w), int(pt1.y * h)
                        x2, y2 = int(pt2.x * w), int(pt2.y * h)

                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        return frame

    def saberClase(self):
        clases = []
        if self.results and self.results.hand_landmarks:
            for handedness in self.results.handedness:
                # Accedemos al nombre de la categoría (Left o Right)
                clases.append(handedness[0].category_name)

        mano = ""
        if len(clases) > 0:
            # Invertimos porque procesa la imagen invertida de la cámara (selfie)
            if clases[0] == "Right":
                mano = "izq"
            else:
                mano = "der"
        return mano

    def findPosition(self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmsList = []
        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                myHand = self.results.hand_landmarks[handNo]
                h, w, c = frame.shape
                for id, lm in enumerate(myHand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmsList.append([id, cx, cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = [xmin, ymin, xmax, ymax]

                if draw:
                    cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmsList, bbox
    def findFingerUp(self):
        fingers = []
        if not self.lmsList:
            return fingers
        # Dedo pulgar
        if self.lmsList[self.tipIds[0]][1] > self.lmsList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Resto de los 4 dedos
        for id in range(1, 5):
            if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    def findDistance(self, p1, p2, frame, draw=True, r=15, t=3):
        if not self.lmsList:
            return 0, frame, []

        x1, y1 = self.lmsList[p1][1:]
        x2, y2 = self.lmsList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]



def main():
    ctime = 0
    ptime = 0
    inicioTiempo = time.perf_counter()
    cap = cv2.VideoCapture(camara)
    detector = HandTrackingDynamic()

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        tiempoActual = time.time()
        if not ret:
            break

        frame = detector.findFingers(frame)
        lmsList, bbox = detector.findPosition(frame)

        mano = detector.saberClase()
        # Comienza dibujado

        y, x, c = frame.shape # Estan invertidos los ejes


        generarBombas(frame,x,y,c)
        generarFrutas(frame,x,y,c)

        #cv2.circle(frame, (int(x/2), int(y/2)), 30, (0, 255, 0), cv2.FILLED)



        if lmsList:
            xmin, ymin, xmax, ymax = bbox
            centroide = calcularCentro(ymin, ymax, xmin, xmax)
            h, b = calcularTam(ymin, ymax, xmin, xmax)
            cv2.circle(frame, (xmin, ymin), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (xmax, ymax), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (centroide[0], centroide[1]), 5, (255, 0, 0), cv2.FILLED)

            # ======================                 Colisiones        ==============================================
            global puntuacion
            for i in range(len(listaFrutas)):
                distancia = calcularDistancia2Puntos(centroide[0], centroide[1], listaFrutas[i][0], listaFrutas[i][1])
                if distancia < tamFruta:
                    puntuacion += incrementoFruta
                    del listaFrutas[i]
                    break


            for i in range(len(listaBombas)):
                distancia = calcularDistancia2Puntos(centroide[0], centroide[1], listaBombas[i][0], listaBombas[i][1])
                if distancia < tamBomba:
                    puntuacion -= disminucionFruta
                    del listaBombas[i]
                    break



        #print(listaFrutas)

        ctime = time.time()
        fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0
        ptime = ctime
        fin = time.perf_counter()
        segundos = fin - inicioTiempo

        #print(segundos)
        #print(fps)
        # Para hacer el cerrado de ventana de forma más controlada ('q' para salir)
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, "puntuacion: " + str(puntuacion), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        cv2.putText(frame, "Tiempo: " + str(segundos), (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        cv2.imshow('frame', frame)

        # ================================= ¿Cómo agregarían condición de salida por tiempo?
        tiempoTotal = segundos
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            print("Puntaje final: " + str(puntuacion))
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
