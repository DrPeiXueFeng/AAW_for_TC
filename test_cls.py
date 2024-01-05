from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging# 日志处理
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))# 'C:\Users\Dr.Pei\Desktop\Pointnet2'
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

"""
配置参数：
--normal 
--log_dir pointnet2_cls_msg
"""

# 解析命令行参数
def parse_args():
    '''PARAMETERS'''
    # 建立参数解析对象
    parser = argparse.ArgumentParser('PointNet')
    # 添加属性：给xx实例增加一个aa属性，如xx.add_argument('aa')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_cls_msg', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting [default: 3]')
    # 采用parser对象的parse_args函数获取解析的参数
    return parser.parse_args()

def test(model, loader, num_class=2, vote_num=1):
    mean_correct = []
    class_acc = np.zeros((num_class,3))# (40, 3)
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()# 张量shape都是默认的batch_size，即24
        classifier = model.eval()# 测试时不启用train函数，即不启用BatchNormalization和Dropout
        vote_pool = torch.zeros(target.size()[0],num_class).cuda()
        for _ in range(vote_num):# default=3
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool/vote_num# 求vote_num次数的平均
        pred_choice = pred.data.max(1)[1]# pred_choice的shape：（24）
        for cat in np.unique(target.cpu()):
            # classacc tensor（B）
            # 求类别的accuracy
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    # 求类别的accuracy
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    # 求instance的accuracy（实例的准确度）
    instance_acc = np.mean(mean_correct)# mean_correct list(103);1024/24=102.8333
    return instance_acc, class_acc# 都是浮点数


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data/lidar_net/'
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = 2
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)

    classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))



if __name__ == '__main__':
    args = parse_args()
    main(args)
