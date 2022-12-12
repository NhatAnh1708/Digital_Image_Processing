import cv2 as cv
import numpy as np



class ImageProcessing:
    def __init__(self):
        self.img_path = '../final1.bmp'
        self.img_path_2 = '../final2.jpg'
        self.img_path_3 = '../final3.jpg'
        self.img_path_4 = '../final4.jpg'
    def showinput(self):
        img = cv.imread(self.img_path_2)
        cv.imshow('input',img)
        cv.waitKey(0)
    def run(self):
        print("Bắt đầu xử lý ảnh ....")
    def show(self,img):
        cv.imshow('Output',img)
        cv.waitKey(0)
    # bo loc trung binh 3x3
    def medianfilter(self):
        img = cv.imread(self.img_path,0)
        row,col = img.shape
        img_new = np.zeros([row, col])
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                temp = [img[i - 1, j - 1],
                        img[i - 1, j],
                        img[i - 1, j + 1],
                        img[i, j - 1],
                        img[i, j],
                        img[i, j + 1],
                        img[i + 1, j - 1],
                        img[i + 1, j],
                        img[i + 1, j + 1]]

                temp = sorted(temp)
                img_new[i, j] = temp[4]
        img_new = img_new.astype(np.uint8)
        self.show(img_new)
        #cv.imwrite("medianfilter.bmp",img_new)
    def avegaringfilter(self):
        img = cv.imread(self.img_path,0)
        row,col = img.shape

        #tao matrix[3x3]
        mask = (np.ones([3,3],dtype=int))/9
        img_new = np.zeros([row,col])
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                temp = img[i - 1, j - 1] * mask[0, 0] + img[i - 1, j] * mask[0, 1] + img[i - 1, j + 1] * mask[0, 2] + \
                       img[i, j - 1] * mask[1, 0] + img[i, j] * mask[1, 1] + img[i, j + 1] * mask[1, 2] + img[
                           i + 1, j - 1] * mask[2, 0] + img[i + 1, j] * mask[2, 1] + img[i + 1, j + 1] * mask[2, 2]

                img_new[i, j] = temp

        img_new = img_new.astype(np.uint8)
        self.show(img_new)

    # day la ham test performance
    def morphology(self):
        img = cv.imread(self.img_path,0)
        kernel = np.ones((2, 2), np.uint8)
        kernel1 = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
        opening1 = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel1)
        closing1 = cv.morphologyEx(opening1, cv.MORPH_CLOSE, kernel1)
        self.show(closing1)
    def morphology_fromopentoclose(self,img):
        #img = cv.imread(self.img_path,0)
        kernel = np.ones((3, 3), np.uint8)

        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

        self.show(closing)
        #cv.imwrite('../exer1/fromopentoclose.bmp', closing)
    def sobel(self):
        img = cv.imread(self.img_path_2)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        height, width, depth = img.shape

        hor = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]

        ver = [[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]

        H = int(len(ver))

        for i in range(0, height - H):
            for j in range(0, width - H):
                summ = 0
                for k in range(0, H):
                    for l in range(0, H):
                        summ = summ + hor[k][l] * img[i + k][j + l]  # hor for horizontal edge
                if (summ[0] > 255):
                    summ = [255, 255, 255]
                elif (summ[0] < 0):
                    summ = [0, 0, 0]
                img[i][j] = summ
        kernel = np.ones((3, 3), np.uint8)
        closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
        dilation = cv.dilate(opening, kernel, iterations=1)
        self.show(dilation)
        #cv.imwrite('Sobel_morphology.jpg',dilation)
    def canny(self):
        img = cv.imread(self.img_path_2)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_blur = cv.GaussianBlur(img_gray, (15, 15), 0)
        # img_blur = cv.medianBlur(img_gray,13) # performance k cao

        img_new = cv.Canny(image=img_blur, threshold1=100, threshold2=200)
        self.show(img_new)

    def canny_ex_4(self):
        img = cv.imread(self.img_path_4)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_blur = cv.medianBlur(img_gray,3) # performance k cao

        img_new = cv.Canny(image=img_blur, threshold1=50, threshold2=50)
        self.show(img_new)
        #cv.imwrite('Canny_edge.jpg',img_new)
    def morphology_fromclosetoopen(self,img):
        #img = cv.imread(self.img_path,0)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
        self.show(opening)
        #cv.imwrite('../exer1/fromclosetoopen.bmp', opening)
    def segmentation(self):
        plant = cv.imread('../final3.jpg')
        rgb = cv.cvtColor(plant, cv.COLOR_BGR2RGB)
        key = input("Màu cần phân vùng:")
        if key == "trang":
            mask = cv.inRange(rgb, (191, 192, 190), (255, 255, 255))
            print("Đang xử lý ảnh....")
            imask = mask > 0
            white = np.zeros_like(plant, np.uint8)
            white[imask] = plant[imask]
            self.show(white)

        elif key == "xanh":
            mask = cv.inRange(rgb, (15, 50, 15), (140, 195, 156))
            print("Đang xử lý ảnh....")
            imask = mask > 0  # pic all pixels>0
            green = np.zeros_like(plant, np.uint8)  # create an array of 0s same size as original image
            green[imask] = plant[imask]
            self.show(green)
        elif key == "hong":
            mask = cv.inRange(rgb, (145, 0, 29), (240, 90, 180))
            print("Đang xử lý ảnh....")
            imask = mask > 0  # pic all pixels>0
            pink = np.zeros_like(plant, np.uint8)  # create an array of 0s same size as original image
            pink[imask] = plant[imask]
            self.show(pink)
        elif key == "vang":
            mask = cv.inRange(rgb, (160, 130, 30), (230, 222, 110))
            print("Đang xử lý ảnh....")
            imask = mask > 0  # pic all pixels>0
            yellow = np.zeros_like(plant, np.uint8)  # create an array of 0s same size as original image
            yellow[imask] = plant[imask]
            self.show(yellow)


if __name__ == '__main__':
    ip=ImageProcessing()
    #ip.showinput()
    #ip.sobel()
    #ip.canny()
    #ip.medianfilter()
    #ip.segmentation()
    ip.canny_ex_4()