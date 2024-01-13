import cv2
import socket
import pickle
import logging

class CameraServer:
    def __init__(self, ip='', port=5005, buffer_size=4096):
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.server_socket = None
        self.cap = None

    def start(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Set up TCP server
        TCP_IP = self.ip
        TCP_PORT = self.port
        BUFFER_SIZE = self.buffer_size

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((TCP_IP, TCP_PORT))
        self.server_socket.listen(1)

    def close(self):
        self.server_socket.close()

    def send_frame(self, frame):
        # Compress frame
        conn, addr = self.server_socket.accept()
        if conn.fileno() == -1:
            logging.error('Failed to accept connection')
            return

        ret, compressed_frame = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error('Failed to compress frame')
            return

        # Serialize compressed frame
        data = pickle.dumps(compressed_frame)
        size = len(data)

        # Send size of the frame first
        conn.sendall(size.to_bytes(4, byteorder='big'))

        # Send frame
        conn.sendall(data)