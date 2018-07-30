from easydict import EasyDict as edict

# 001: original mnist
# 002: original stanford dogs

model_params = edict()

model_params.emb_dim = {'001': 2,
                        '002': 1024}

model_params.m = {'001': 8,
                  '002': 12}

model_params.d = {'001': 8,
                  '002': 4}

model_params.k = {'001': 3,
                  '002': 3}

model_params.alpha = {'001': 1.0,
                      '002': 1.0}