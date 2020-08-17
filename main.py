import argparse
import yaml
from pathlib import Path

from logging import getLogger

from tensorboardX import SummaryWriter
from torchsummary import summary

from utils.path_process import Paths
from utils.setup_logger import setup_logger

logger = getLogger(__name__)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', type=str, default='./configs/default.yml')
    parser.add_argument('--inference', action='store_true', default=False)
    args = parser.parse_args()
    return args

def main(args):

    with open(args.configfile) as f:
        configs = yaml.safe_load(f)

    ## path process (path definition, make directories)
    now = datetime.now().isoformat()
    log_dir = Path(configs['log_dir']) / now
    paths = Paths(log_dir=log_dir)

    ### setup logs and summary writer ###
    setup_logger(logfile=paths.logfile)

    writer = SummaryWriter(str(paths.summary_dir))

    ### setup GPU or CPU ###
    if configs['n_gpus'] > 0 and torch.cuda.is_available():
        logger.info('CUDA is available! using GPU...\n')
        device = torch.device('cuda')
    else:
        logger.info('using CPU...\n')
        device = torch.device('cpu')

    ### Dataset ###
    logger.info('preparing dataset...')

    ### DataLoader ###


    ### Network ###
    logger.info('preparing network...')

    ### Visualize Results ###

    ### Train or Inference ###
    

if __name__ == "__main__":
    args = parser()
    main(args)