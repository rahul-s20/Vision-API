from flask import Blueprint, jsonify, make_response, request
from controllers.ai_response_controller import response_main
from utils.web_tokenization import token_required
from controllers.auth_controller import is_authenticated

ai_response_ = Blueprint('ai_response_', __name__, url_prefix='/api/v1/vision')


@ai_response_.route('/test', methods=['GET'])
def user_test():
    return make_response(jsonify({'success': True, 'data': 'working'}), 200)


@ai_response_.route('/response', methods=['POST'])
def vision_response():
    body = request.get_json(silent=True)
    res = response_main(input_sentence=body['ask'])
    return make_response(jsonify({'success': True, 'data': res}), 200)