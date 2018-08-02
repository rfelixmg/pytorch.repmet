from easydict import EasyDict as edict

# 001: original mnist
# 002: original stanford dogs

model_params = edict()

model_params.emb_dim = {'001': 2,
                        '002': 64,
                        '003': 2}#1024}

model_params.dataset = {'001': 'MNIST',
                        '002': 'STANDOGS',
                        '003': 'OXFLOWERS'}

# m = num clusters per minibatch
model_params.m = {'001': 8,
                  '002': 12,
                  '003': 12}

# d = num of samples retrieved per cluster
model_params.d = {'001': 8,
                  '002': 4,
                  '003': 4}

# k = num of modes per class
model_params.k = {'001': 3,
                  '002': 3,
                  '003': 3}

model_params.alpha = {'001': 1.0,
                      '002': 0.71,
                      '003': 1.0}#2.43}

model_params.lr = {'001': 1e-4,
                   '002': 0.00292,
                   '003': 1e-4}#0.024}