import os
import requests
import argparse

def download_file_from_google_drive(id, destination):
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

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = {'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id' : id, 'confirm' : token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def download_file_from_server(server, subset, destination):
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

    session = requests.Session()
    if server == 'google':
        URL = "https://docs.google.com/uc?export=download"
        params = {'id': ids[subset]}
    elif server == 'snu':
        URL = 'http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/' + subset + '.zip'
        params = {}

    response = session.get(URL, params=params, stream=True)
    token = get_confirm_token(response)
    if token:
        params['confirm'] = token
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

parser = argparse.ArgumentParser(description='Download REDS dataset from google drive to current folder', allow_abbrev=False)

parser.add_argument('--server', type=str, default='google', choices=('google', 'snu'), help='download server choice.')
parser.add_argument('--all', action=True, help='download full REDS dataset')
parser.add_argument('--train_sharp', action='store_true', help='download train_sharp.zip')
parser.add_argument('--train_blur', action='store_true', help='download train_blur.zip')
parser.add_argument('--train_blur_comp', action='store_true', help='download train_blur_comp.zip')
parser.add_argument('--train_sharp_bicubic', action='store_true', help='download train_sharp_bicubic.zip')
parser.add_argument('--train_blur_bicubic', action='store_true', help='download train_blur_bicubic.zip')
parser.add_argument('--val_sharp', action='store_true', help='download val_sharp.zip')
parser.add_argument('--val_blur', action='store_true', help='download val_blur.zip')
parser.add_argument('--val_blur_comp', action='store_true', help='download val_blur_comp.zip')
parser.add_argument('--val_sharp_bicubic', action='store_true', help='download val_sharp_bicubic.zip')
parser.add_argument('--val_blur_bicubic', action='store_true', help='download val_blur_bicubic.zip')
parser.add_argument('--test_blur', action='store_true', help='download test_blur.zip')
parser.add_argument('--test_blur_comp', action='store_true', help='download test_blur_comp.zip')
parser.add_argument('--test_sharp_bicubic', action='store_true', help='download test_sharp_bicubic.zip')
parser.add_argument('--test_blur_bicubic', action='store_true', help='download test_blur_bicubic.zip')

args = parser.parse_args()

ids = { 'train_sharp': '1YLksKtMhd2mWyVSkvhDaDLWSc1qYNCz-',
        'train_blur': '1Be2cgzuuXibcqAuJekDgvHq4MLYkCgR8',
        'train_blur_comp': '1hi6348BB9QQFqVx2PY7pKn32HQM89CJ1',
        'train_sharp_bicubic': '1a4PrjqT-hShvY9IyJm3sPF0ZaXyrCozR',
        'train_blur_bicubic': '10u8gthv2Q95RMCb1LeCN8N4ozB8TVjMt',
        'val_sharp': '1MGeObVQ1-Z29f-myDP7-8c3u0_xECKXq',
        'val_blur': '1N8z2yD0GDWmh6U4d4EADERtcUgDzGrHx',
        'val_blur_comp': '13d1uzqLdbsQzeZkWgdF5QVHqDSjfE4zZ',
        'val_sharp_bicubic': '1sChhtzN9Css10gX7Xsmc2JaC-2Pzco6a',
        'val_blur_bicubic': '1i3NAb7EmF4fCYadGaHK54-Zgx9lIC2Gp',
        'test_blur': '1dr0--ZBKqr4P1M8lek6JKD1Vd6bhhrZT',
        'test_blur_comp': '1OctyKR3ER_YWrZxKxQsZzLis3BvLSOFO',
        'test_sharp_bicubic': '1y0Jle6xB41TdRK_QMJ_E8W_iBMxwq_Rh',
        'test_blur_bicubic': '14YszfzUAeAfwP0ZA2FRzAiVxxZLg7-tY',
        }

# Download files in REDS directory
if os.path.basename(os.getcwd()) == 'REDS':
    root_dir = '.'
else:
    os.makedirs('REDS', exist_ok=True)
    root_dir = 'REDS'

for subset in  :
    argdict = args.__dict__
    if args.all or argdict[subset]:
        filename = '{}/{}.zip'.format(root_dir, subset)
        servername = 'Google Drive' if args.server == 'google' else 'SNU CVLab'
        print('Downloading {}.zip from {}'.format(subset, servername))
        download_file_from_server(args.server, subset, filename)    # download the designated subset