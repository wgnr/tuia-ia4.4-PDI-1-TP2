# PARTE A -  DETECCION DE PATENTES

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Problema_2_A(img):
    img_original=img.copy()
    # UMBRALADO
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (37, 8))
    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # FILTRAMOS POR COLOR HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img, (0, 0, 101), (180,10,182))
    img = cv2.bitwise_and(img, img, mask=img_mask)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clau_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,3))
    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, clau_ker)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    aux_rois=[]
    for idx in range(1, nlabels):
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]
        if 537 < area < 2349 and 1.739 < w/h < 3.7:
            aux_rois.append(((x, y), (x + w, y + h)))
        
    assert aux_rois!=[], "NO SE ENCONTRO NINGUNA PATENTE"

    if len(aux_rois)==1:
        # rois.append(aux_rois[0])
        return aux_rois[0]

    results=[]
    for (X,Y),(dX,dY) in aux_rois:
        img_aux=img_original.copy()
        img_aux=img_aux[Y:dY,X:dX]
        img_aux=cv2.cvtColor(img_aux, cv2.COLOR_BGR2GRAY)
        area=img_aux.shape[0]*img_aux.shape[1]
        n_white_pixels = np.sum(img_aux > 200)/area 
        results.append(n_white_pixels)

    return  aux_rois[np.argmin(results)]



if __name__ == "__main__":
    imgs = [cv2.imread(f"img/img{i:02n}.png") for i in range(1, 13)]
    fig, axs = plt.subplots(3, 4, figsize=(14, 14))

    for i, img in enumerate(imgs):
        roi = Problema_2_A(img)
        (X,Y),(dX,dY) = roi
        cv2.rectangle(imgs[i], (X, Y), (dX,dY), 150, 1)
        axs[i // 4, i % 4].cla()
        axs[i // 4, i % 4].imshow(imgs[i], cmap="gray")
        axs[i // 4, i % 4].set_title(i + 1)
    
    plt.show()
    