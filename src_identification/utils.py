import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch


def str2bool(v):
    """
    Function to transform strings into booleans.

    v: string variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_best_models(net, optimizer, output_path, best_records, epoch, acc, acc_cls, cm):
    if len(best_records) < 5:
        best_records.append({'epoch': epoch, 'acc': acc, 'acc_cls': acc_cls, 'cm': cm})

        torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))
    else:
        # find min saved acc
        min_index = 0
        for i, r in enumerate(best_records):
            if best_records[min_index]['acc_cls'] > best_records[i]['acc_cls']:
                min_index = i
        # check if currect acc is greater than min saved acc
        if acc_cls > best_records[min_index]['acc_cls']:
            # if it is, delete previous files
            min_step = str(best_records[min_index]['epoch'])

            os.remove(os.path.join(output_path, 'model_' + min_step + '.pth'))
            os.remove(os.path.join(output_path, 'opt_' + min_step + '.pth'))

            # replace min value with current
            best_records[min_index] = {'epoch': epoch, 'acc': acc, 'acc_cls': acc_cls, 'cm': cm}

            # save current model
            torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))
