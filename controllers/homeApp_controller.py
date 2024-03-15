import requests
from configuration.device_config import all_devices
from flask import jsonify, make_response


def switch(device: str, mode: str, url: str = 'http://192.168.1.22'):
    """
           segment = 25(light) / 27(fan)
           is_on = on / off
           """
    session = requests.Session()
    try:
        pin_ = all_devices[device][0]
        alias = all_devices[device][1]
        if mode == 'on' or mode == 'off':
            print(f'{url}/{alias}?state={mode}&out_put={pin_}')
            res = session.get(f'{url}/{alias}?state={mode}&out_put={pin_}')

            return make_response(jsonify({'success': True, 'data': {"device": device,"state": mode}}), 200)
        else:
            return make_response(jsonify({'success': False, 'data': "Please prvide the correct mode"}), 400)
    except Exception as er:
        return make_response(jsonify({'success': False, 'data': f"{er}"}), 400)

