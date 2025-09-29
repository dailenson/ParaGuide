import time
import os

""" prepare logdir for tensorboard and logging output"""
def set_log(args):
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_logs = f"{args.saved}/{args.exp}-{t}/logs"
    save_models = f"{args.saved}/{args.exp}-{t}/models"
    os.makedirs(save_logs, exist_ok=True)
    os.makedirs(save_models, exist_ok=True)
    return save_logs, save_models