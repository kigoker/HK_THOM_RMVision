import serial
import time

def open_serial(port, baud_rate):
    """打开串口并配置基本参数"""
    try:
        ser = serial.Serial(port, baud_rate, timeout=1,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            bytesize=serial.EIGHTBITS)
        if ser.is_open:
            print(f"Serial port {port} opened successfully")
        return ser
    except Exception as e:
        print(f"Failed to open serial port: {e}")
        return None

def read_from_serial(ser):
    """从串口读取数据"""
    try:
        data = ser.readline()  # 读取一行数据
        if data:
            print(f"Received: {data.decode().strip()}")
    except Exception as e:
        print(f"Failed to read data: {e}")

def write_to_serial(ser, data):
    """向串口发送数据"""
    try:
        ser.write(data.encode())
        print(f"Sent: {data}")
    except Exception as e:
        print(f"Failed to send data: {e}")

def UART_init():
    port = "/dev/ttyUSB0"
    baud_rate = 115200

    # 打开串口
    ser = open_serial(port, baud_rate)
    return ser

def main():
    port = "/dev/ttyUSB0"
    baud_rate = 115200

    # 打开串口
    ser = open_serial(port, baud_rate)
    if ser and ser.is_open:
        try:
            # 循环接收和发送数据
            while True:
                read_from_serial(ser)
                time.sleep(1)
                write_to_serial(ser, "A")
                time.sleep(1)
        finally:
            ser.close()
            print("Serial port closed")

if __name__ == "__main__":
    main()
