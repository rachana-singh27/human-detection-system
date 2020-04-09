import csv
import cv2
import os
import numpy as np
#import sys
#sys.path.append("")


k=0
l=0
path1='C:/Users/Rachana Singh/PycharmProjects/pos osu/'
path2='C:/Users/Rachana Singh/PycharmProjects/neg osu/'
dir='C:/Users/Rachana Singh/PycharmProjects/data/'
count_pos=[]
count_neg=[]
for subdir, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith(".csv"):
            file1=open((os.path.join(subdir,file)),"r")
            reader = csv.reader(file1, delimiter=',')
            pos_list = []
            neg_list = []
            pos_label = []
            neg_label= []
            for row in reader:
                if row[0].startswith('img'):
                    print(row[0], row[1])  # number of boxes
                    img=cv2.imread(os.path.join(subdir, row[0]),0)

                    #norm=np.linalg.norm(img)                        #normalization
                    #img=img/norm
                    #print(img.dtype)
                    #print('img.shape',img.shape)
                    #cv2.imshow('win1', img[:, :, 0].astype(np.uint8))
                    #cv2.waitKey(0)
                    # print(img.shape)
                    list1 = []
                    for i in range(int(row[1])):
                        x_min = int(row[4 * i + 2])
                        y_min = int(row[4 * i + 3])
                        x_max = int(row[4 * i + 4])
                        y_max = int(row[4 * i + 5])
                        center_x = int((x_max + x_min) / 2)
                        center_y = int((y_max + y_min) / 2)
                        # print(x_min,y_min,x_max,y_max)
                        # print(center_x,center_y)
                        list1.append([center_x, center_y])
                        # x=img[center_y-15:center_y+15,center_x-10:center_x+10].copy()


                        if center_y - 16 >= 0 and center_y + 16 <= img.shape[0] and center_x - 8 >= 0 and center_x + 8 <= img.shape[1]:
                            x = img[center_y - 16:center_y + 16, center_x - 8:center_x + 8,].copy()
                            print(x)
                            k+=1
                            filename='savedpos'+str(k)+'.jpg'
                            cv2.imwrite(path1+filename,x)
                        if center_y + 16 <= img.shape[0] and center_y + 48 <= img.shape[0] and center_x + 9 <=img.shape[1] and center_x + 25 <= img.shape[1]:
                            y1 = img[center_y + 16:center_y + 48, center_x + 9:center_x + 25, ].copy()
                            print(y1)
                            l+=1
                            filename = 'savedneg'+str(l)+'.jpg'
                            cv2.imwrite(path2 + filename, y1)
                        elif center_y -48>=0 and center_y - 16>=0 and center_x - 25>=0 and center_x -9>=0:
                            y2 = img[center_y -48:center_y - 16, center_x - 25:center_x -9, ].copy()
                            print(y2)
                            l+=1
                            filename = 'savedneg'+str(l)+'.jpg'
                            cv2.imwrite(path2+filename,y2)
