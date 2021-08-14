import struct
import pickle
import socket

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def decode_msg(msg):
    res = pickle.loads(msg)
    return res

def encode_msg(data):
    msg = pickle.dumps(data)
    return msg

class clientCommunication():
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def send_msg(self, msg):
        msg = encode_msg(msg)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))
        send_msg(self.s, msg)

    def receive_msg(self):
        received = recv_msg(self.s)
        received = decode_msg(received)
        return received

    def close_channel(self):
        self.s.close()

class serverCommunication():
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.host, self.port))
        self.s.listen()

    def send_msg(self, conn, msg):
        msg = encode_msg(msg)
        send_msg(conn, msg)

    def receive_msg(self, conn):
        received = recv_msg(conn)
        received = decode_msg(received)
        return received

    def accept_conn(self):
        conn, addr = self.s.accept()
        return conn, addr

    def close_channel(self):
        self.s.close()
