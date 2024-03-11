from flask import Blueprint, jsonify, make_response, request
from controllers.user_controller import create_user, user_login, user_details
from utils.web_tokenization import token_required
from controllers.auth_controller import is_authenticated

user_ = Blueprint('user_', __name__, url_prefix='/api/v1/user')


@user_.route('/test', methods=['GET'])
def user_test():
    return make_response(jsonify({'success': True, 'data': 'working'}), 200)


@user_.route('/register', methods=['POST'])
def register_user():
    body = request.get_json(silent=True)
    cu = create_user(data=body)
    return cu


@user_.route('/login', methods=['POST'])
def sign_user():
    body = request.get_json(silent=True)
    res = user_login(data=body)
    return res


@user_.route('/userDetails', methods=['GET'])
@token_required
def get_user_details():
    token = request.headers.get('Authorization')
    res = user_details(token=token)
    return res


@user_.route('/isAuthenticated', methods=['GET'])
def is_auth():
    token = request.headers.get('Authorization') if request.headers.get('Authorization') else ''
    res = is_authenticated(token=token)
    return res
