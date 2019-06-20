import socket

port = 7777
sock = None
sock_state = 0

def create_socket():
    '''
        create a sock
    '''
    global sock, port, sock_state

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', port))
    except socket.error as msg:
        print("[ERROR] %s\n"%(msg))
        sock_state = -1
    print('create socket successful')

def close_socket():
    '''
        close the socket
    '''
    global sock
    if (sock is not None):
        sock.close()
        sock = None
    print('close the socket successful')

def send(string):
    '''
        send the data with TCP sock.
        data format: "data_type data[0] data[1] ..."
    '''
    global sock, sock_state
    if(sock_state != -1):
        if sock is None:
            create_socket()
        else:
            sock.sendall(string.encode())

if(__name__ == '__main__'):
    create_socket()
    send('0 hello world')
    close_socket()