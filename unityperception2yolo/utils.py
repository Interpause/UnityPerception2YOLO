from datetime import datetime


def get_formatted_time():
    return datetime.today().strftime("%Y-%m-%d-%H%M")
