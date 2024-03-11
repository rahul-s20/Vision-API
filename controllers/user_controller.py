from models.users import Users
from utils.helpers import strict_ip_filer, hashing, check_hash, cookie_expiration_set
from flask import make_response, jsonify
from utils.web_tokenization import encode_token, decode_token
from bson import ObjectId


def create_user(data: dict):
    try:
        user_obj = Users()
        data = strict_ip_filer(ip_data=data, fields={'email', 'name', 'password'})
        find_one = Users.objects.filter(email=data['email']).first()
        if find_one:
            return make_response(jsonify({'success': False, 'data': 'The user already exists. Please try with another '
                                                                    'email id'}))
        else:
            user_obj.name = data['name']
            user_obj.email = data['email']
            user_obj.password = hashing(content=data['password'])
            user_obj.save()
            return make_response(jsonify({'success': True, 'data': {
                "id": str(user_obj.id),
                "name": user_obj.name,
                "email": user_obj.email,
                "privilege": user_obj.privilege
            }}), 200)
    except Exception as er:
        return make_response(jsonify({'success': False, 'data': f"{er}"}), 400)


def user_login(data: dict):
    filtered_data = strict_ip_filer(data, {'email', 'password'})
    find_one = Users.objects.filter(email=filtered_data['email']).first()

    if find_one:
        check_pass = check_hash(content=data['password'], hashed_content=find_one.password)
        if check_pass:
            token, options = encode_token(user_id=str(find_one.id))
            response = make_response(jsonify({'success': True, 'data': {
                "id": str(find_one.id),
                "name": find_one.name,
                "email": find_one.email,
                "privilege": find_one.privilege,
                'token': token
            }}), 200)
            response.set_cookie('token', token, expires=cookie_expiration_set(), domain=None,
                                secure=True, httponly=True, samesite='none')
            return response
        else:
            return make_response(jsonify({'success': False, 'data': "Invalid password"}), 401)
    elif not filtered_data['email'] or not filtered_data['password']:
        return make_response(jsonify({'success': False, 'data': "Missing fields"}), 400)
    else:
        return make_response(jsonify({'success': False, 'data': "Please register yourself"}), 401)


def user_details(token: str):
    decoded_obj = decode_token(token=token)
    user_id = decoded_obj['sub']
    find_one = Users.objects.filter(id=ObjectId(user_id)).first()
    if find_one:
        return make_response(jsonify({'success': True, 'data': {
            "id": str(find_one.id),
            "name": find_one.name,
            "email": find_one.email,
        }}), 200)
    else:
        return make_response(jsonify({'success': False, 'data': "User not found"}), 400)
