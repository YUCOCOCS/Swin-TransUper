import argparse
import logging
import os
import random
import importlib
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_isic import isic_loader
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume, get_num_parameters,test_single_volume_isic
from config import get_config
from model.swin_transuper import SwinTransUper
from scipy.ndimage.morphology import binary_fill_holes, binary_opening
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str,  
                    default='swin_224_7_2level_epoch_189.pth', help='absolute path to saved ckpt during training.')
parser.add_argument('--config_file', type=str,
                    default='swin_224_7_3level', help='config file name w/o suffix')
parser.add_argument('--volume_path', type=str,
                    default='./ISIC2017', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='isic2017', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', default='./results/', type=str, help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
# if args.dataset == "isic2017":
#     args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)


def inference(args, model, test_save_path=None):
    #db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir) # 加载数据
    db_test = isic_loader(path_Data = args.volume_path, train = False,Test=True) # 验证集上进行加载
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0) #加载测试集的数据
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt=[]
    predictions = []
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name,img = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'],sampled_batch['copy']
        image = image.to(device, dtype=torch.float)
        pred = model(image)
        msk_pred = torch.sigmoid(pred)
        gt.append(label.numpy()[0,0])
        msk_pred = msk_pred.cpu().detach().numpy()[0, 0]
        msk_pred  = np.where(msk_pred>=0.5, 1, 0)
        msk_pred = binary_opening(msk_pred, structure=np.ones((6,6))).astype(msk_pred.dtype)
        msk_pred = binary_fill_holes(msk_pred, structure=np.ones((6,6))).astype(msk_pred.dtype)
        msk_pred[msk_pred==1]=255
        #img = cv2.imdecode(msk_pred, cv2.IMREAD_COLOR)
        #img = Image.fromarray(msk_pred)
        img = img.squeeze(0).numpy()
        label[label==1] = 255
        img = Image.fromarray(np.uint8(img))
        img.save("/home/JianjianYin/transdeeplab/ours/image/"+str(case_name)+".jpg")
        #img.save("/home/JianjianYin/transdeeplab/Predict/"+str(case_name)+".jpg")
        #cv2.imwrite("/home/JianjianYin/transdeeplab/image/"+str(case_name)+".jpg",img)
        cv2.imwrite("/home/JianjianYin/transdeeplab/ours/Predict/"+str(case_name)+".jpg",msk_pred)
        cv2.imwrite("/home/JianjianYin/transdeeplab/ours/label/"+str(case_name)+".jpg",label.cpu().numpy()[0,0])

       # predictions.append(msk_pred) 
    # predictions = np.array(predictions)
    # gt = np.array(gt)

    # y_scores = predictions.reshape(-1)
    # y_true   = gt.reshape(-1)

    # y_scores2 = np.where(y_scores>0.5, 1, 0)
    # y_true2   = np.where(y_true>0.5, 1, 0)

    # #F1 score
    # F1_score = f1_score(y_true2, y_scores2, labels=None, average='binary', sample_weight=None)
    # print ("\nF1 score (F-measure) or DSC: " +str(F1_score))
    # confusion = confusion_matrix(np.int32(y_true), y_scores2)
    # print (confusion)
    # accuracy = 0
    # if float(np.sum(confusion))!=0:
    #     accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    # print ("Accuracy: " +str(accuracy))
    # specificity = 0
    # if float(confusion[0,0]+confusion[0,1])!=0:
    #     specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    # print ("Specificity: " +str(specificity))
    # sensitivity = 0
    # if float(confusion[1,1]+confusion[1,0])!=0:
    #     sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    # print ("Sensitivity: " +str(sensitivity))

    #logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f  mean_SE:%f  mean_SP: %f  mean_ACC:%f' % (performance, mean_hd95,mean_se,mean_sp,mean_acc))
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'isic2017': {
            'Dataset': isic_loader,
            'volume_path': args.volume_path,
            'list_dir': None,
            'num_classes': 1,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']# 1
    args.is_pretrain = True

    
    model_config = importlib.import_module(f'model.configs.{args.config_file}') #导入对应的配置文件 
    args.img_size = model_config.EncoderConfig.img_size # important and bug friendly!  224 * 224
    
    net = SwinDeepLab(
        model_config.EncoderConfig, 
        model_config.ASPPConfig, 
        model_config.DecoderConfig
    ).cuda() # 导入配置文件
    
    # Printing out the number of parameters in the model and each module
    encoder_p, aspp_p, decoder_p = get_num_parameters(net) # 获得各个部分的参数量 是以百万为单位的
    print(f"Number of Encoder Parameters: {encoder_p:.3f}")
    print(f"Number of ASPP Parameters: {aspp_p:.3f}")
    print(f"Number of Decoder Parameters: {decoder_p:.3f}")
    print(f"Total Number of Parameters: {encoder_p + aspp_p + decoder_p:.3f}")

    
    snapshot = args.ckpt_path.split('/')[-1]
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    #for i in range(170,200):
    args.ckpt_path = "isic2017/swin_224_7_3level_epoch_178.pth"
    logging.info("此时加载%s模型"%(args.ckpt_path))
    msg = net.load_state_dict(torch.load(args.ckpt_path)) # 加载训练好的模型
    inference(args, net, test_save_path) #进行推理


