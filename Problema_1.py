import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread("img/monedas.jpg")
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplica operación de dilatación para cerrar los bordes
kernel_dilatacion = np.ones((10, 10), np.uint8)


def dilate_image(img, iterations=1):
    return cv2.dilate(img, kernel_dilatacion, iterations=iterations)


# Canny
imagen_suavizada = cv2.medianBlur(imagen_gris, 5)
canny = cv2.Canny(imagen_suavizada, 50, 150)

bordes_dilatados = dilate_image(canny, iterations=3)

# Aplicar erosión para eliminar pequeñas imperfecciones
test = cv2.erode(
    bordes_dilatados,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    iterations=1,
)

# Encuentra los contornos en la imagen con bordes definidos
contornos, _ = cv2.findContours(test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Crear una máscara en blanco del mismo tamaño que la imagen original
mascara = np.zeros_like(imagen_gris)
# Dibuja los contornos en la máscara
cv2.drawContours(mascara, contornos, -1, (255), thickness=cv2.FILLED)

(
    componentes_conectadas,
    etiquetas,
    estadisticas,
    centroides,
) = cv2.connectedComponentsWithStats(mascara, cv2.CV_32S, connectivity=8)

imagen_resultado = np.zeros_like(test)

for i in range(1, componentes_conectadas):
    area = estadisticas[i, cv2.CC_STAT_AREA]

    if area > 600:
        mascara = np.uint8(etiquetas == i)
        contorno, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(imagen_resultado, contorno, -1, 120, 2)


# Funcion para rellenar los contornos
def fill_contours(img):
    seed_point = (0, 0)
    blanco = (255, 255, 255)

    flags = 4
    lo_diff = (10, 10, 10)
    up_diff = (10, 10, 10)

    cv2.floodFill(img, None, seed_point, blanco, lo_diff, up_diff, flags)
    return ~img


imagen_resultado = fill_contours(imagen_resultado)

result = cv2.dilate(
    imagen_resultado, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=5
)

(
    componentes_conectadas,
    etiquetas,
    estadisticas,
    centroides,
) = cv2.connectedComponentsWithStats(result, cv2.CV_32S, connectivity=8)

squares_masks = []
aux = np.zeros_like(result)
labeled_image = cv2.merge([aux, aux, aux])

RHO_TH = 0.83

for i in range(1, componentes_conectadas):
    mascara = np.uint8(etiquetas == i)
    contorno, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = cv2.contourArea(contorno[0])
    perimetro = cv2.arcLength(contorno[0], True)
    rho = 4 * np.pi * area / (perimetro**2)
    flag_circ = rho > RHO_TH
    if flag_circ:
        if area > 123000:
            labeled_image[mascara == 1, 0] = 255
        elif area > 85000:
            labeled_image[mascara == 1, 1] = 255
        else:
            labeled_image[mascara == 1, 2] = 255
    else:
        labeled_image[mascara == 1, 2] = 120
        labeled_image[mascara == 1, 1] = 120

        square_mask = np.zeros_like(result)
        cv2.drawContours(square_mask, contorno, -1, 120, 2)
        squares_masks.append(square_mask)

dst = cv2.addWeighted(imagen, 0.7, labeled_image, 0.3, 0)

RHO_TH = 0.78

for id, square_mask in enumerate(squares_masks):
    square_mask = fill_contours(square_mask)
    squares = cv2.bitwise_and(canny, canny, mask=square_mask)

    squares_dilatados = dilate_image(squares)

    (
        componentes_conectadas,
        etiquetas,
        estadisticas,
        centroides,
    ) = cv2.connectedComponentsWithStats(squares_dilatados, cv2.CV_32S, connectivity=8)
    count = 0
    for i in range(1, componentes_conectadas):
        area = estadisticas[i, cv2.CC_STAT_AREA]
        mascara = np.uint8(etiquetas == i)
        contorno, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if area < 10000 and area > 500:
            area = cv2.contourArea(contorno[0])
            perimetro = cv2.arcLength(contorno[0], True)
            rho = 4 * np.pi * area / (perimetro**2)
            flag_circ = rho > RHO_TH
            if flag_circ:
                count += 1
    print(f"Dado {id + 1} tiene: {count}")

# Mostramos resultado
plt.figure(figsize=(11, 11))
plt.imshow(dst)
plt.title("Imagen con los contornos")
plt.axis("off")
plt.show()
