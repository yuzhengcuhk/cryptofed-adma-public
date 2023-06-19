import socket
import hashlib
import os

#向client发送数据，对应client的get
#发送pth格式文件和publickey

bob_worker = socket.socket()
bob_worker.bind(('localhost', 8080))
bob_worker.listen()
while True:
    conn, addr = bob_worker.accept()
    print('等待指令：')
    while True:
        data = conn.recv(1024)
        if not data:
            print('客户端断开')
            break

        #第一次接收的是命令，包括get和文件名，用filename接收文件名
        cmd, filename = data.decode().split()

        #从接收到的文件名判断是不是一个文件
        if os.path.isfile(filename):
            f = open(filename, 'rb')  #如果是，读模式打开这个文件
            m = hashlib.md5()         #生成md5对象
            file_size = os.stat(filename).st_size  #将文件大小赋值给file_size
            conn.send(str(file_size).encode())     #发送文件大小
            conn.recv(1024)           #接收确认信息

            for line in f:            #开始发文件
                m.update(line)        #发一行更新一下md5值
                conn.send(line)       #一行一行发送

            #print('file md5:', m.hexdigest())  打印整个文件的md5
            f.close()
            conn.send(m.hexdigest().encode())   #send md5
        print('send done')
    break
bob_worker.close()
