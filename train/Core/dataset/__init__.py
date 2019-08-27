from . import irmas

def feeder(path, gpu):
    return irmas.Feeder(path, gpu)
