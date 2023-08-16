import os


def dir_create(path, recursive=True):
    path = os.path.expanduser(path)
    path = os.path.dirname(path)
    does_exist = os.path.exists(path)
    if not does_exist:
        if recursive:
            os.makedirs(path)
        else:
            os.mkdir(path)
    return None



