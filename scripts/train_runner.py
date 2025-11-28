import torch
import argparse
from trainer import Trainer
import warnings
import os

class Train_Runner:

    def __init__(self,args):

        # 1) Setup parameters ----------- :
        self.setup_params(args)

        # 2) Setup Device ----------- :
        if args.force_cpu:
            device = torch.device("cpu")
            print("强制使用CPU进行训练")
        elif args.force_gpu or args.force_incompatible_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA不可用，无法强制使用GPU训练")
            device = torch.device("cuda:0")
            print(f"强制使用GPU: {torch.cuda.get_device_name(0)} 进行训练")
            if args.force_incompatible_gpu:
                print("警告：正在强制使用可能不兼容的GPU")
        else:
            # 默认行为：如果有兼容的CUDA设备则使用，否则使用CPU
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                print(f"使用GPU: {torch.cuda.get_device_name(0)} 进行训练")
            else:
                device = torch.device("cpu")
                print("使用CPU进行训练")

        # 3) Load in Model ----------- :
        from network.TEDS_Net import TEDS_Net as net
        net = net(self.params)
        net.to(device)

        # 4) Train and Evalte the Model ---------- :
        trainer = Trainer(self.params, device, net)
        trainer.dothetraining()
        trainer.do_evalutation()


    def setup_params(self,args):
        """ _set up the training parameters from default options provided_

        Args:
            args (module): contains the settings for training
        """
        
        if args.dataset =="mnist":
            from parameters.mnist_parameters import Parameters
        elif args.dataset=="ACDC":
            from parameters.acdc_parameters import Parameters

        self.params = Parameters.from_dict({'data':args.dataset})
        
        # 修正数据路径，使其指向正确的数据库目录
        if args.dataset=="ACDC":
            # 获取项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # 设置正确的数据路径
            data_path = os.path.join(project_root, 'Resources', 'database')
            self.params.dataset.datapath = data_path
            print(f"数据路径已设置为: {data_path}")


if __name__ == '__main__':
    """ MKWyburd GitHub TEDS-Net
    
    The network described in MICCAI 2021 for myocaridum segmentation using the ACDC dataset are stored in parameters/acdc_parameters. The prior shape generator is in the dataloader folder

    To test the code, you can use the simple MNIST example, using Pytorch automated MNIST dataset. The data will be downloaded into a "tmp" folder. 
    """

    parser = argparse.ArgumentParser(
        description='Run TEDS-Net Segmentation')

    parser.add_argument('--dataset', 
                        help = 'Which dataset we are using',
                        choices=['ACDC','mnist'],
                        default='mnist')
    parser.add_argument('--force_gpu', 
                        help = 'Force training on GPU (will raise error if CUDA is not available)',
                        action='store_true')
                        
    parser.add_argument('--force_incompatible_gpu', 
                        help = 'Force training on GPU even if it may be incompatible',
                        action='store_true')

    args = parser.parse_args()

    # Perform our Training:
    Train_Runner(args)