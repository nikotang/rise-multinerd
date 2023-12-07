import os

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def load_conf(path):
    '''Loads the configuration .yml file. '''
    with open(path) as fin:
        conf_txt = fin.read()
    conf = yaml.load(conf_txt, Loader=Loader)
    assert 'raw_yaml' not in conf
    conf['raw_yaml'] = conf_txt
    
    return conf