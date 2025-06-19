import cv2
import numpy as np
import math

 
def main():
    capture = cv2.VideoCapture(0)
    if capture.isOpened() is False:
        print("Error opening the camera")
        return
    while capture.isOpened():
        ret, frame = capture.read()

        if ret is True:
            # 在图像上绘制检测结果
            blue, g, r = cv2.split(frame)  # 分离通道，在opencv中图片的存储通道为BGR非RBG
            # 绘制轮廓  是在原图像上进行画轮廓
            ret2, binary = cv2.threshold(blue, 220, 255, 0)
            Gaussian = cv2.GaussianBlur(binary, (5, 5), 0)  # 高斯滤波
            # edge = cv2.Canny(binary, 50, 150)  # 边缘检测
            draw_img = Gaussian.copy()
            whole_h, whole_w = binary.shape[:2]
            # 输出的第一个值为图像，第二个值为轮廓信息，第三个为层级信息
            contours, hierarchy = cv2.findContours(image=draw_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            contours = list(contours)
            contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
            width_array = []
            height_array = []
            point_array = []
            for cont in contours[:5]:
                x, y, w, h = cv2.boundingRect(cont)
                try:
                    if h / w >= 2 and h / whole_h > 0.1 and h > w:
                        # if height / h > 0.05:
                        width_array.append(w)
                        height_array.append(h)
                        point_array.append([x, y])
                except:
                    continue
            point_near = [0, 0]
            min = 10000
            for i in range(len(width_array) - 1):
                for j in range(i + 1, len(width_array)):
                    value = abs(width_array[i] * height_array[i] - width_array[j] * height_array[j])
                    if value < min:
                        min = value
                        point_near[0] = i
                        point_near[1] = j
            try:
                rectangle1 = point_array[point_near[0]]
                rectangle2 = point_array[point_near[1]]
                point1 = [rectangle1[0] + width_array[point_near[0]] / 2, rectangle1[1]]
                point2 = [rectangle1[0] + width_array[point_near[0]] / 2, rectangle1[1] + height_array[point_near[0]]]
                point3 = [rectangle2[0] + width_array[point_near[1]] / 2, rectangle2[1]]
                point4 = [rectangle2[0] + width_array[point_near[1]] / 2, rectangle2[1] + height_array[point_near[1]]]
                print(point1, point2, point3, point4)
                x = np.array([point1, point2, point4, point3], np.int32)
                box = x.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(frame, [box], True, (0, 255, 0), 2)
            except:
                continue
                # 显示处理后的图像
            cv2.imshow('name', frame)
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 释放资源
    capture.release()
    cv2.destroyAllWindows()

def img_show(name, src):
    cv2.imshow(name, src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def NUO():
    video = cv2.VideoCapture(0)
    while True:
        ret, img = video.read()
        blue, g, r = cv2.split(img)  # 分离通道，在opencv中图片的存储通道为BGR非RBG
        # 绘制轮廓  是在原图像上进行画轮廓
        ret2, binary = cv2.threshold(blue, 220, 255, 0)
        Gaussian = cv2.GaussianBlur(binary, (5, 5), 0)  # 高斯滤波
        # edge = cv2.Canny(binary, 50, 150)  # 边缘检测
        draw_img = Gaussian.copy()
        whole_h, whole_w = binary.shape[:2]
        # 输出的第一个值为图像，第二个值为轮廓信息，第三个为层级信息
        contours, hierarchy = cv2.findContours(image=draw_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        contours = list(contours)
        contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
        width_array = []
        height_array = []
        point_array = []
        for cont in contours[:5]:
            x, y, w, h = cv2.boundingRect(cont)
            try:
                if h / w >= 2 and h / whole_h > 0.1 and h > w:
                    # if height / h > 0.05:
                    width_array.append(w)
                    height_array.append(h)
                    point_array.append([x, y])
            except:
                continue
        point_near = [0, 0]
        min = 10000
        for i in range(len(width_array) - 1):
            for j in range(i + 1, len(width_array)):
                value = abs(width_array[i] * height_array[i] - width_array[j] * height_array[j])
                if value < min:
                    min = value
                    point_near[0] = i
                    point_near[1] = j
        try:
            rectangle1 = point_array[point_near[0]]
            rectangle2 = point_array[point_near[1]]
            point1 = [rectangle1[0] + width_array[point_near[0]] / 2, rectangle1[1]]
            point2 = [rectangle1[0] + width_array[point_near[0]] / 2, rectangle1[1] + height_array[point_near[0]]]
            point3 = [rectangle2[0] + width_array[point_near[1]] / 2, rectangle2[1]]
            point4 = [rectangle2[0] + width_array[point_near[1]] / 2, rectangle2[1] + height_array[point_near[1]]]
            print(point1, point2, point3, point4)
            x = np.array([point1, point2, point4, point3], np.int32)
            box = x.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [box], True, (0, 255, 0), 2)
        except:
            continue
        cv2.imshow('name', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放资源
    video.release()
    cv2.destroyAllWindows()

if __name__ =="__main__":
    main()