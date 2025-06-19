# -*- coding:utf-8 -*-
import threading
import time

cancel_tmr = False

def start():
	#具体任务执行内容
    print("hello world")

def heart_beat():
	# 打印当前时间
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    if not cancel_tmr:
        start()
        # 每隔3秒执行一次
        threading.Timer(3, heart_beat).start()
if __name__ == '__main__':
    heart_beat()
    # 15秒后停止定时器
    time.sleep(15)
    cancel_tmr = True
