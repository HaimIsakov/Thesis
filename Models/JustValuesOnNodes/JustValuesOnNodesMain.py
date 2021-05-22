import json
import csv
import logging
import nni
from JustValuesOnNodesClass import *
from dataset_class_just_values_on_nodes import *
from torch.utils.data import DataLoader, random_split

LOG = logging.getLogger('nni_logger')

try:
    RECEIVED_PARAMS = json.load(open('params_file.json', 'r'))
    data_file_path = "taxonomy_gdm_file.csv"
    tag_file_path = "tag_gdm_file.csv"
    dataset = JustValuesOnNodesDatasetClass(data_file_path, tag_file_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    samples_len = len(dataset)
    data_size = dataset.get_vector_size()
    epochs = 10

    # calculate lengths off train and dev according to split ~ (0,1)
    len_train = int(samples_len * 0.675)
    len_dev = int(samples_len * 0.125)
    len_test = samples_len - len_train - len_dev

    train, validate, test = random_split(dataset, [len_train, len_dev, len_test])
    batch_size = RECEIVED_PARAMS['batch_size']

    # set train loader
    train_loader = DataLoader(
        train.dataset,
        batch_size=batch_size,
        shuffle=True
    )
    count_zeros, count_ones = dataset.count_each_class()
    loss_weights = [1 / count_zeros, 1 / count_ones]
    # set validation loader
    dev_loader = DataLoader(
        validate.dataset,
        batch_size=batch_size
    )

    # set test loader
    test_loader = DataLoader(
        test.dataset,
        batch_size=batch_size,
    )

    model = JustValuesOnNodes(data_size, RECEIVED_PARAMS)
    model = model.to(device)

    my_train(model, RECEIVED_PARAMS, epochs, train_loader, device, loss_weights)
    # auc
    auc = my_test(model, test_loader, loss_weights, device)

    # report final result
    LOG.debug('Final result is: %d', auc)
    nni.report_final_result(auc)

except Exception as e:
    LOG.exception(e)
    raise
