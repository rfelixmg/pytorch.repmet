from easydict import EasyDict as edict

from os.path import join, expanduser
import platform

general = edict()
general.paths = edict()

# Set this to the main directory for your PC
if platform.node() == 'HUB':  # UNI computer
    general.paths.root = '/media/hayden/Storage21/'
elif platform.node() == 'HUB-HOME':  # HOME computer
    general.paths.root = '/media/hayden/UStorage/'
elif 'adelaide.edu.au' in platform.node():
    general.paths.root = '/fast/users/a1211517/'
else:
    general.paths.root = expanduser("~")

general.paths.graphing = join(general.paths.root, 'MODELS', 'repmet', 'plots')
general.paths.models = join(general.paths.root, 'MODELS', 'repmet', 'saves')
general.paths.imagesets = join(general.paths.root, 'DATASETS', 'IMAGE')
general.paths.videosets = join(general.paths.root, 'DATASETS', 'VIDEO')