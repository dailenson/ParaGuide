import os
import time
from datetime import datetime, timezone
import torch
from torchvision.utils import make_grid, save_image
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from iam_dataset import *
from model import *
from tensorboardX import SummaryWriter

torch.manual_seed(1)

if __name__ =='__main__':

    import argparse
    parser = argparse.ArgumentParser('Dual Triplet')
    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--dataset', type=str, nargs=2, default='./../../../DATASETS/BHSig260/Bengali')
    parser.add_argument('--saved', type=str, default='./saved')
    parser.add_argument('--load_model', type=str, default='./../Final models/[FINAL_PAR-Encoder]BHSig260_Bengali_SSL_Encoder_RN18.pth')
    parser.add_argument('--batchsize', type=int, default=64)  # change for Testing; default=16
    parser.add_argument('--print_freq', type=int,default=100)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--lambd', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--round', type=int, choices=[1,2,3,4,5])
    parser.add_argument('--ssl_bkb', type=bool, default=False, help='if backbone is a SSL model or not')
    parser.add_argument("--bkb", type=str) # choices=['PAR', 'SimCLR','Barlow','SimSiam'])
    parser.add_argument('--set', type=str, default='self', choices=['self', 'cross_fulldata'])
    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--size', nargs=2, type=int, default=[512, 512])
    # parser.add_argument('--backbone', type=str, default='projector')
    args = parser.parse_args()

    # bkb = args.load_model.split('/')[2].split('_')[0] if args.ssl_bkb is True else "PAR"
    bkb =  args.bkb    

    "initialize saved directories"
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_logs = f"{args.saved}/{args.exp}-{t}/logs"
    save_models = f"{args.saved}/{args.exp}-{t}/models"
    os.makedirs(save_logs, exist_ok=True)
    os.makedirs(save_models, exist_ok=True)

    print("\n--------------------------------------------------\n")    
    print(args)

    train_loader, _, _, train_dset = get_dataloader(args)

    print('-'* 50)


    ### setting up tensorboard ###
    # dt_curr = datetime.now(timezone.utc).strftime("%b:%d_%H:%M:%S")
    # tfb_dir = os.path.join(os.getcwd(), f"{args.saved_models}/tboard_logs/BHSig260Bengali_{EXPT}-_{dt_curr}")
    # Vis = Visualizer(tfb_dir)

    model = Triplet_Model(args)
    # model.encoder.load_state_dict(torch.load(args.load_model))
    epoch = 0

    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        model.encoder.load_state_dict(checkpoint, strict=True)
        print(f">> Resume training from pre-trained encoder model")
    tb_summary = SummaryWriter(log_dir=save_logs)
    model = model.to(device)

    logs = {'epoch':[], 'step':[], 'epoch_loss':[]}

    # 3. train model
    # for epoch in range(args.max_epochs):
    while epoch != args.max_epochs: ## Retraining from checkpoint ##
        epoch_loss = 0.0
        model.scheduler.step()
        for ii, batch_train in enumerate(train_loader):
            step_count = (ii+1)+epoch*len(train_loader)
            start = time.time()
            model.train()
            nce_loss_guidance = model.train_iter(batch=batch_train)
            tb_summary.add_scalar('nce_loss_guidance', nce_loss_guidance, step_count)
            nce_loss = nce_loss_guidance
            print(f'Epoch: {epoch+1}/{args.max_epochs} |' \
                    f'Step: {ii+1}/{len(train_loader)} |' \
                    f'nce_loss_guidance: {(nce_loss_guidance):.8f} |' \
                    f'time: {(time.time()-start):.8f} sec')
            
            epoch_loss += nce_loss 

            if (ii+1) % args.print_freq == 0:
                logs['epoch'].append(epoch+1)
                logs['step'].append(ii+1)
                logs['epoch_loss'].append(epoch_loss/(ii+1))
                pd.DataFrame.from_dict(logs).to_csv(f"{save_logs}/IAM_nobg_R={args.round}.csv", index=False)
    
        epoch += 1

        # save model after every 5 epochs
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(),f'{save_models}/DTL_epoch={epoch+1}.pt')
    print('Training complete !!')