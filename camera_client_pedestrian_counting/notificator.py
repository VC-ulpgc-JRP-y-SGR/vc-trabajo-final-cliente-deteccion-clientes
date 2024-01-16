
import requests

ip = "192.168.8.102:5000"

def notify_client_entered():
    requests.post('http://'+ip+'/client_entered/')

def notify_client_leave():
    requests.post('http://'+ ip + '/client_exited/')