from mongoengine import Document, StringField, BooleanField
from utils.helpers import current_date


class Users(Document):
    email = StringField(required=True, unique=True)
    name = StringField(max_length=30, min_length=4, required=True)
    password = StringField(min_length=4, required=True)
    privilege = StringField(max_length=50, default='low')
    timestamp = StringField(max_length=10, default=current_date())
    last_updated = StringField(max_length=10, default=current_date())
    status = StringField(max_length=10, default='offline')
    otp = StringField(max_length=8, default='')
    verified = BooleanField(default=False)
    is_active = BooleanField(default=True)