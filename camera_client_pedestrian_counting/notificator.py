
import requests

def notify_client_entered():
    requests.post('http://127.0.0.1:5000/client_entered/')

def notify_client_leave():
   requests.post('http://127.0.0.1:5000/client_exited/')

notify_client_leave()