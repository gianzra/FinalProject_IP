import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#import data
rgb = cv.imread('daun.jpeg')
img = cv.imread('daun.jpeg',0)

#1. operasi sobel derivative X untuk mendeteksi garis tepi horizontal
sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
abs_sobelx64f = np.absolute(sobelx64f)
sobelx_8u = np.uint8(abs_sobelx64f)

#2. operasi sobel derivative y untuk mendeteksi garis tepi vertikal
sobely64f = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
abs_sobely64f = np.absolute(sobely64f)
sobely_8u = np.uint8(abs_sobely64f)

#3. operasi magnitude -> Nilai Absolute sobel X + Nilai Absolute Sobel Y
magnitudesobel = cv.magnitude(sobelx64f,sobely64f)
abs_sobel64f = np.absolute(magnitudesobel)
sobel_8u = np.uint8(abs_sobel64f)

#4. melakukan pengisisan kontur garis tepi
# mengkonversi pixel ke grayscale dan membaca ukuran pixel
hh, ww = img.shape[:2]

#5. melakukan operasi thresholding agar dapat meningkatkan ketegasan dari tepi
thresh = cv.threshold(sobel_8u, 30, 255, cv.THRESH_BINARY)[1]

#6.mendapatkan garis tepi dengan memilih kontur yang paling luas
contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv.contourArea)

#7. memperbarui isi piksel kontur terluas menjadi putih untuk dijadikan mask
mask = np.zeros_like(img)
cv.drawContours(mask, [big_contour], 0, (255,255,255), cv.FILLED)

#8. melakukan operasi bitwise untuk menggabungkan antara mask dengan gambar yang asli
res = cv.bitwise_and(rgb,rgb, mask= mask)

plt.subplot(3,2,1),plt.imshow(rgb,cmap = 'gray')
plt.title('1.1.Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,2),plt.imshow(res,cmap = 'gray')
plt.title('2.1.Result->Edge Detection'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,5),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('1.2.Magnitude'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,6),plt.imshow(mask,cmap = 'gray')
plt.title('2.2Mask'), plt.xticks([]), plt.yticks([])
plt.show()

cv.imshow('1.1.Original', rgb)
cv.imshow('1.2.Magnitude', sobel_8u)
cv.imshow('2.2Mask', mask)
cv.imshow('2.1.Result->Edge Detection', res)

cv.waitKey(0)
cv.destroyAllWindows()