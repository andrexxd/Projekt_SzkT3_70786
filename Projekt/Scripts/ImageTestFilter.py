import cv2
import numpy as np


img = cv2.imread("test.png")

# 1. Skala szarości
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 2. Negatyw
inverted = cv2.bitwise_not(gray)
# 3. Rozmycie negatywu (im większy kernel, tym grubsze linie)
blurred = cv2.GaussianBlur(inverted, (95, 93), sigmaX=0, sigmaY=0)
# 4. Rozjaśnianie (color dodge) – podzielenie szarości przez odwrócony rozmyty
#    Używamy dzielenia float, aby uniknąć obcięcia wartości
sketch = cv2.divide(gray, 255 - blurred, scale=256.0)
# 5. Konwersja do 8-bit i na 3 kanały BGR
sketch_8u = np.uint8(sketch)
final_sketch = cv2.cvtColor(sketch_8u, cv2.COLOR_GRAY2BGR)


# cv2.imshow("Image", img)
cv2.imshow("Edges", final_sketch)
cv2.waitKey(0)
cv2.destroyAllWindows()
