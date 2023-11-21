# PARTE A -  DETECCION DE PATENTES
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Problema_2_A import Problema_2_A


fig, axs = plt.subplots(3, 4, figsize=(14, 14))
def Problema_2_B(img_patente):
    rois=[]
    img = cv2.cvtColor(img_patente, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 12))
    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    # UMBRALADO

    img_mask = cv2.inRange(img, 104, 255)
    img = cv2.bitwise_and(img, img, mask=img_mask)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img, (0, 0, 73), (180,255,231))
    img = cv2.bitwise_and(img, img, mask=img_mask)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

    for idx in range(1, nlabels):
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]
        if 12 < area < 200 and h/w > 1:
            rois.append(((x, y), (x + w, y + h)))
            
    return rois



if __name__ == "__main__":
    imgs = [cv2.imread(f"img/img{i:02n}.png") for i in range(1, 13)]

    for i, img in enumerate(imgs):
        patente_roi = Problema_2_A(img)
        (X,Y),(dX,dY) = patente_roi
        rois = Problema_2_B(imgs[i][Y:dY,X:dX])
        axs[i // 4, i % 4].cla()
        for letra_roi in rois:
            (x,y),(dx,dy) = letra_roi
            cv2.rectangle(imgs[i], (X+x, Y+y), (X+dx,Y+dy), 150, 1)
        axs[i // 4, i % 4].imshow(img, cmap="gray")
        axs[i // 4, i % 4].set_title(i + 1)
    plt.show()
