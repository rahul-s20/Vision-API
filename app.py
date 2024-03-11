from flask import Flask
from flask_mongoengine import MongoEngine
from routes.user_route import user_
from routes.ai_response_route import ai_response_

db = MongoEngine()


def create_app(env='Development'):
    app = Flask(__name__)
    app.config.from_object('configuration.config.%s' % env)
    db.init_app(app)
    app.register_blueprint(user_)
    app.register_blueprint(ai_response_)
    return app
