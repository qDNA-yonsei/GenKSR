from datetime import datetime as dt


def timestamp():
    return dt.now().strftime('%d-%m-%Y %H:%M:%S')
