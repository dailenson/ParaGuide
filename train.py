import time
import torch
import warnings
warnings.filterwarnings('ignore')
from data_loader.loader import *
from models.model import *
from utils.logger import set_log
from utils.util import fix_seed
from tensorboardX import SummaryWriter

if __name__ =='__main__':

    import argparse
    parser = argparse.ArgumentParser('ParaGuide')
    parser.add_argument('--base_dir', type=str, default='./data/ForensicsIAM')
    parser.add_argument('--dataset', type=str, nargs=2, default=["IAM", "VATr"])
    parser.add_argument('--saved', type=str, default='./saved')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--size', nargs=2, type=int, default=[512, 512])

    args = parser.parse_args()

    #fix the random seed for reproducibility
    random_seed = 1
    fix_seed(random_seed)
    
    #prepare log file
    save_logs, save_models = set_log(args)
    print("\n--------------------------------------------------\n")    
    print(args)
    
    # set dataset
    train_loader, _, _, train_dataset = get_dataloader(args)

    print('-'* 50)

    # build model, criterion and optimizer
    model = ParaGuide(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    #load checkpoint
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint, strict=True)
        print(f">> Resume training from pre-trained model")

    logs = {'epoch':[], 'step':[], 'epoch_loss':[]}

    # start training iterations 
    epoch = 0
    tb_summary = SummaryWriter(log_dir=save_logs)
    while epoch != args.max_epochs:
        epoch_loss = 0.0
        model.scheduler.step()
        for ii, batch_train in enumerate(train_loader):
            step_count = (ii+1)+epoch*len(train_loader)
            start = time.time()
            model.train()
            nce_loss_aux, nce_loss_guidance = model.train_iter(batch=batch_train)
            tb_summary.add_scalar('nce_loss_aux', nce_loss_aux, step_count)
            tb_summary.add_scalar('nce_loss_guidance', nce_loss_guidance, step_count)
            nce_loss = nce_loss_aux + nce_loss_guidance
            print(f'Epoch: {epoch+1}/{args.max_epochs} |' \
                    f'Step: {ii+1}/{len(train_loader)} |' \
                    f'nce_loss_aux: {(nce_loss_aux):.8f}/ |' \
                    f'nce_loss_guidance: {(nce_loss_guidance):.8f} |' \
                    f'time: {(time.time()-start):.8f} sec')
    
        epoch += 1

        # save model weights after every 5 epochs
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(),f'{save_models}/epoch={epoch+1}.pt')
    print('Training complete !!')