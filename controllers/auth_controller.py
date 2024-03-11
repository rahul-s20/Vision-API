from models.users import Users
from utils.helpers import strict_ip_filer, hashing, check_hash, cookie_expiration_set
from flask import make_response, jsonify
from utils.web_tokenization import encode_token, decode_token
from bson import ObjectId


def is_authenticated(token: str):
    try:
        decoded_obj = decode_token(token=token)
        user_id = decoded_obj['sub']
        find_one = Users.objects.filter(id=ObjectId(user_id)).first()
        if find_one:
            return make_response(jsonify({'success': True, 'data': True}), 200)
        else:
            return make_response(jsonify({'success': False, 'data': False}), 401)
    except Exception as er:
        return make_response(jsonify({'success': False, 'data': False}), 401)
