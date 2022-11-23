import cv2

src = "pangeranDiponegoro.png"

img = cv2.imread(src)

# 1. melakukan konversi gambar original menjadi grayscale
img_gray = cv2.imread(src, 0)

# 2. Menginisiasi nilai tengah dari grayscale
t = 128
H, W = img_gray.shape[:2]
img_binary = img_gray.copy()

# 3. melakukan operasi konversi dari pixel ke biner
for i in range(H):
    for j in range(W):
        if img_binary[i, j] >= t: #pixel lebih dari atau sama dengan 128 = putih
            img_binary[i, j] = 255
        elif img_binary[i, j] < t:  # pixel kurang dari atau sama dengan 128 = hitam
            img_binary[i, j] = 0

cv2.imshow("original", img)
cv2.imshow("binary", img_binary)

cv2.waitKey(0)
cv2.destroyAllWindows();