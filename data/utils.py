import torch

def get_inputs(dataset, indexs):
    """
    Gets the input data from a dataset
    :param dataset: The dataset
    :param indexs: List of the sample indexs
    :return: A tensor with the inputs stacked
    """
    inputs = None
    c = 0
    for index in indexs:
        if c == 0:
            inputs = torch.unsqueeze(dataset[index][0], 0)
        else:
            inputs = torch.cat((inputs, torch.unsqueeze(dataset[index][0], 0)), 0)
        c += 1
    return inputs