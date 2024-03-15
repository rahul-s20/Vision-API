from flask import Blueprint, jsonify, make_response, request
from utils.web_tokenization import token_required
from controllers.auth_controller import is_authenticated
from controllers.homeApp_controller import switch

vision_home_ = Blueprint('vision_home_', __name__, url_prefix='/api/v1/VisionHome')


@vision_home_.route('/switch', methods=['GET', 'POST'])
def switch_():
    device = request.args.get('device')
    mode = request.args.get('mode')
    res = switch(device=device, mode=mode)
    return res