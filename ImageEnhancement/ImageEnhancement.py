import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('covid_image.png', 0)
img_mask = cv.imread('mask.png', 0)

# 1. bitwise masking + rgb img
#    -Operasi bitwise untuk menggabungkan antara gambar mask dan covid image(First Output)
restore_bitwise = cv.bitwise_and(img,img, mask= img_mask)

# 2. Reduction Noise
#     Traverse the image. For every 3X3 area,
#    -Noise Reduction dengan median filter 3x3 (Second Output)
noise_img = restore_bitwise
m, n = noise_img.shape
medianfilter = np.zeros([m, n])
for i in range(1, m-1):
    for j in range(1, n-1):
        temp = [noise_img[i-1, j-1],
               noise_img[i-1, j],
               noise_img[i-1, j + 1],
               noise_img[i, j-1],
               noise_img[i, j],
               noise_img[i, j + 1],
               noise_img[i + 1, j-1],
               noise_img[i + 1, j],
               noise_img[i + 1, j + 1]]
        temp = sorted(temp)
        medianfilter[i, j]= temp[4]
    medianfilter = medianfilter.astype(np.uint8)

# 3. repairing contrast
#    - Perbaikan kontras dengan historgam equalization (Third Output)
before_equal = medianfilter.astype('uint8')
after_equal = cv.equalizeHist(medianfilter.astype('uint8'))

#real pict
plt.subplot(1, 3, 1),plt.hist(before_equal.ravel(),256,[0,256], color = 'r');

#equal
# 4. Finishing, Melakukan threshold atau thresholding
plt.subplot(1, 3, 2),plt.hist(after_equal.ravel(),256,[0,256], color = 'g');
ret, thresh_img = cv.threshold(after_equal, 120, 255, cv.THRESH_TOZERO)

# Perbaikan kontras dengan histogram equalization
img_before_equ = thresh_img.astype('uint8')
res_contrast = cv.equalizeHist(thresh_img.astype('uint8'))

# output program
cv.imshow('Bitwise', restore_bitwise)
cv.imshow('Median Filtering', medianfilter)
cv.imshow('Equalized Histogram',after_equal)
cv.imshow('Threshold To zero', thresh_img)

# saving file
cv.imwrite('res_bitwise.jpg', restore_bitwise)
cv.imwrite('res_contrast.jpg', medianfilter)
cv.imwrite('final.jpg', after_equal)
cv.imwrite('res_noise_removal.jpg', thresh_img)

cv.waitKey(0)
cv.destroyAllWindows()