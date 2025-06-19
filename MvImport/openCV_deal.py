import image_change
import os
import sys
import numpy as np
from os import getcwd
import cv2
import ctypes
from ctypes import byref
import platform
import RM_serial
import threading
import time
import struct

from enum import Enum

xmidnum = 0
ymidnum = 0
DELAT_MAX = 30

Width = 1280
Height = 1024
Fire_flag = 0
Direction = [Width/2 , Height/2]
Direction_err = [0 , 0]

Fire_flager = 0

Fire_color = "red"    #设置 blue 则攻击蓝色方 ，设置 red 则攻击红色方  00发送失败

class AdjustMode(Enum):
    WIDTH_GREATER_THAN_HEIGHT = 1
    ANGLE_TO_UP = 2

ZJB = {
    'center': (0, 0),
    'size': (0, 0),
    'angle': 0.0
}

def adjust_rec(rec, mode):
    width = rec[1][0]
    height = rec[1][1]
    angle = rec[2]

    if mode == 1:
        if width < height:
            width, height = height, width
            angle += 90.0

    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0

    if mode == 2:
        if angle >= 45.0:
            width, height = height, width
            angle -= 90.0
        elif angle < -45.0:
            width, height = height, width
            angle += 90.0

    return (rec[0], (width, height), angle)


def color_identifier(img, point1=None, point2=None):
    """
    在图像中识别指定区域的主导颜色（红/蓝）
    
    参数:
        img (np.ndarray): BGR格式图像帧
        point1 (tuple, 可选): 左上角坐标 (x1, y1)
        point2 (tuple, 可选): 右下角坐标 (x2, y2)
        
    返回:
        str: "red" 或 "blue" 或 None（未选择区域时）
    """
    # 坐标有效性校验
    if (point1 is None) != (point2 is None):
        raise ValueError("必须同时提供point1和point2坐标")
    
    if point1 is None:
        # 启动交互式ROI选择
        def on_mouse(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                points = [(x, y)]
            elif event == cv2.EVENT_LBUTTONUP and len(points) == 1:
                points = (points[0], (x, y))
        
        points = None
        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", on_mouse)
        cv2.imshow("Select ROI", img.copy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # 自动修正坐标顺序
        x1, y1 = point1
        x2, y2 = point2
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        points = ( (x1, y1), (x2, y2) )

    if points is None:
        return None

    # 提取ROI区域
    (x1, y1), (x2, y2) = points
    if x1 >= x2 or y1 >= y2:
        raise ValueError("坐标点无效：左上角坐标必须小于右下角坐标")

    roi_img = img[y1:y2, x1:x2]

    # 颜色识别逻辑（保持不变）
    hsv_lower_red = np.array([0, 50, 50])
    hsv_upper_red = np.array([10, 255, 255])
    hsv_lower_blue = np.array([100, 50, 50])
    hsv_upper_blue = np.array([130, 255, 255])

    hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv_roi, hsv_lower_red, hsv_upper_red)
    mask_blue = cv2.inRange(hsv_roi, hsv_lower_blue, hsv_upper_blue)

    red_pixels = np.sum(mask_red)
    blue_pixels = np.sum(mask_blue)

    threshold = 0.5 * (red_pixels + blue_pixels)
    
    if red_pixels > threshold:
        return "red"
    elif blue_pixels > threshold:
        return "blue"
    else:
        return "unknown"
    
def clamp(value, center, delta_max):
    """
    将数值限制在中心值的±delta_max范围内
    
    参数:
        value (float): 输入值
        center (float): 中心基准值
        delta_max (float): 最大允许偏移量
        
    返回:
        float: 被限制后的值
    """
    min_val = center - delta_max
    max_val = center + delta_max
    return max(min_val, min(value, max_val))

def Light_strip_judgement(Light_Contour, image):
    vc = []
    vRec = []
    global Fire_flager
    Fire_flager = 0
    for i in range(len(Light_Contour)):
        # 求轮廓面积
        Light_Contour_Area = cv2.contourArea(Light_Contour[i])
        # 去除较小轮廓&fitEllipse的限制条件
        if Light_Contour_Area < 15 or len(Light_Contour[i]) <= 10:
            continue
        # 用椭圆拟合区域得到外接矩形
        Light_Rec = cv2.fitEllipse(Light_Contour[i])
        Light_Rec = adjust_rec(Light_Rec, 2)
        
        # 将 Light_Rec 转换为字典或自定义对象
        light_info = {
            'center': Light_Rec[0],
            'size': Light_Rec[1],
            'angle': Light_Rec[2]
        }
        
        if light_info['angle'] > 10:  # Light_Rec[2] 是角度
            continue
        
        # 长宽比和轮廓面积比限制
        if light_info['size'][0] / light_info['size'][1] > 1.5 or Light_Contour_Area / (light_info['size'][0] * light_info['size'][1]) < 0.5:
            continue
        
        vc.append(light_info)  # 将字典添加到 vc 列表中

    for i in range(len(vc)):
        for j in range(i + 1, len(vc)):
            # 判断是否为相同灯条
            Contour_angle = abs(vc[i]['angle'] - vc[j]['angle'])  # 角度差
            if Contour_angle >= 7:  #原来是7度
                continue
            # 长度差比率
            Contour_Len1 = abs(vc[i]['size'][1] - vc[j]['size'][1]) / max(vc[i]['size'][1], vc[j]['size'][1])
            # 宽度差比率
            Contour_Len2 = abs(vc[i]['size'][0] - vc[j]['size'][0]) / max(vc[i]['size'][0], vc[j]['size'][0])
            if Contour_Len1 > 0.25 or Contour_Len2 > 0.7:
                continue

            # 计算中心点
            ZJB['center'] = {
                'x': (vc[i]['center'][0] + vc[j]['center'][0]) / 2,
                'y': (vc[i]['center'][1] + vc[j]['center'][1]) / 2
            }

            # 使用 clamp 函数替代 filter
            ZJB['center']['x'] = clamp(ZJB['center']['x'], xmidnum, DELAT_MAX)
            ZJB['center']['y'] = clamp(ZJB['center']['y'], ymidnum, DELAT_MAX)

            # 计算角度
            ZJB['angle'] = (vc[i]['angle'] + vc[j]['angle']) / 2

            # 计算高度和宽度
            nh = (vc[i]['size'][1] + vc[j]['size'][1]) / 2  # 高度
            nw = np.sqrt((vc[i]['center'][0] - vc[j]['center'][0]) ** 2 + 
                        (vc[i]['center'][1] - vc[j]['center'][1]) ** 2)  # 宽度

            # 更新 ZJB 的大小
            ZJB['size'] = (nw, nh)
            vRec.append(ZJB)
            point1 = (int(vc[i]['center'][0]), int(vc[i]['center'][1] + 20))  # 确保是整数
            point2 = (int(vc[j]['center'][0]), int(vc[j]['center'][1] - 20))  # 确保是整数

            color_detect = color_identifier(image, point1, point2)

            print(color_detect)
            cv2.rectangle(image, point1, point2, (0, 120, 255), 2)  # 将装甲板框起来
            if color_detect == Fire_color :
                Direction[0] = (point1[0]+point2[0])/2
                Direction[1] = (point1[1]+point2[1])/2
                Fire_flager = 1
                #print((Direction[0],Direction[1]))
                Direction_err[0] = Width/2 - Direction[0]
                Direction_err[1] = Height/2 -Direction[1] 
                #print((Direction_err[0],Direction_err[1]))
                cv2.circle(image, (int(ZJB['center']['x']), int(ZJB['center']['y'])), 10, (0, 120, 255))  # 在装甲板中心画一个圆
            else:
                Fire_flager= 0
                Direction[0], Direction[1] = Width / 2, Height / 2  # 默认中点
                Direction_err[0], Direction_err[1] = 0, 0  # 差值归零    
    # 如果没有检测到符合的装甲板，重置方向差值
    if not vRec:
        Fire_flager= 0
        Direction[0], Direction[1] = Width / 2, Height / 2  # 默认中点
        Direction_err[0], Direction_err[1] = 0, 0  # 差值归零    
        
    return 



def openCV_dealwith(image):
     # 彩色图转灰度图
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 显示灰度图，调试用
    cv2.resize(gray_img, (0, 0), fx=0.5, fy=0.5, dst=gray_img)
    #cv2.imshow("gray", gray_img)
    cv2.resize(gray_img, (0, 0), fx=2, fy=2, dst=gray_img)

    # 进行二值化
    binary_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)[1]

    # 显示二值图，调试用
    cv2.resize(binary_img, (0, 0), fx=0.5, fy=0.5, dst=binary_img)
    #cv2.imshow("binary_img", binary_img)
    cv2.resize(binary_img, (0, 0), fx=2, fy=2, dst=binary_img)

    # 获取轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    Light_strip_judgement(contours ,image)
    # 获取旋转矩形并显示
    rotated_rects = []
    for contour in contours:
        rotated_rect = cv2.minAreaRect(contour)
        rotated_rects.append(rotated_rect)

    drawrect = image.copy()
    for rotated_rect in rotated_rects:
        points = cv2.boxPoints(rotated_rect)
        points = np.int0(points)
        #center = calculate_center(points)
        #print(center)
        cv2.polylines(drawrect, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # 显示制的结果
    cv2.resize(drawrect, (0, 0), fx=0.5, fy=0.5, dst=drawrect)
    cv2.imshow("drawrect", drawrect)

cancel_tmr = False

def start():
	#具体任务执行内容
    print("hello world")
# 定义发送数据包的函数
def send_packet(data1, data2, data3):
    # 包头和包尾
    packet_header = b'\xFE'
    packet_footer = b'\xFF'
    #0复数 1正数
    W_L_flag = b'\x00'
    W_R_flag = b'\x01'
    H_L_flag = b'\x00'
    H_H_flag = b'\x01'
    # 打包3个整型数据（4字节无符号整数，小端字节序）
    if data1>=0 and data2 >=0 :
        packet_data = struct.pack('<iii', data1, data2, data3)
        w_flag = W_R_flag
        h_flag = H_H_flag
    elif data1>=0 and data2 < 0 :
        packet_data = struct.pack('<iii', data1, -data2, data3)
        w_flag = W_R_flag
        h_flag = H_L_flag
    elif data1<0 and data2 >= 0 :
        packet_data = struct.pack('<iii', -data1, data2, data3)
        w_flag = W_L_flag
        h_flag = H_H_flag
    else :
        packet_data = struct.pack('<iii', -data1, -data2, data3)
        w_flag = W_L_flag
        h_flag = H_L_flag

    # 计算校验和（仅针对中间数据部分）
    checksum = sum(packet_data) & 0xFF

    # 构造完整数据包
    packet = packet_header + packet_data  +w_flag + h_flag + packet_footer

    # 发送数据包
    ser.write(packet)

    # 打印发送的内容（调试用）
    # print(f"Sent packet: {[hex(byte) for byte in packet]}")

def Timer_one():
	# 打印当前时间
    global Fire_flag
    #print(time.strftime('%Y-%m-%d %H:%M:%S'))
    #RM_serial.write_to_serial(ser,int(Direction[0]))
    #RM_serial.write_to_serial(ser,int(Direction[1]))
    if Direction[0]<=Width/2 + Width/10 and Direction[0] >=Width/2-Width/10 and Direction[1] >=Height/2-Height/10 and Direction[1]<= Height/2+Height/10 and Fire_flager ==1 :
        Fire_flag = 1
    else:
        Fire_flag = 0 
    send_packet(int(Direction_err[0]), int(Direction_err[1]), Fire_flag)

    #向下发送的数据
    print((Direction[0],Direction[1]))
    print((Direction_err[0],Direction_err[1]),Fire_flag)
    #send_packet(int(Direction[0]), int(Direction[1]), Fire_flag)
    if not cancel_tmr:
        # 每隔3秒执行一次
        threading.Timer(0.01, Timer_one).start()
ser = RM_serial.UART_init()
def main():
    #Init Uart
    
    # 枚举设备
    deviceList = image_change.enum_devices(device=0, device_way=False)
    # 判断不同类型设备
    image_change.identify_different_devices(deviceList)
    cam, stDeviceList = image_change.creat_camera(deviceList, 0, log=False)
    image_change.open_device(cam)
    # 开启设备取流
    image_change.start_grab_and_get_data_size(cam)
    Timer_one()
    while True :
        # 主动取流方式抓取图像
        
        openCV_data = image_change.access_get_image(cam, active_way="getImagebuffer")
        openCV_dealwith(openCV_data)
        #RM_serial.write_to_serial(ser,"data")
    # 关闭设备与销毁句柄
    image_change.close_and_destroy_device(cam)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main()