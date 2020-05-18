import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

file_id = '19W9P7V5AF1uRYmh0zThlRexolI-NeEFu'
destination = './weights/vgg16+VB_224x224.pt'
download_file_from_google_drive(file_id, destination)

file_id = '1-A26Vdo6vGUEilAuDuuQZd1e8kWm1xir'
destination = './weights/vgg16+1conv_224x224_e5.pt'
download_file_from_google_drive(file_id, destination)