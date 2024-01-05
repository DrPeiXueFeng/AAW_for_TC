# AAW_for_TC
Leveraging the powerful capabilities of PointNet++ and a custom PCD, we propose an effective method for PCD classification on construction sites and implement proactive collision avoidance for tower cranes based on this approach. 
# Requirments
python 3.8
pytorch 2.0.1+cuda 118
numpy 1.24.1
# Run
## Check model in ./models   
## e.g pointnet2_cls_msg
python train_partseg.py --model pointnet2_cls_msg  --log_dir pointnet2_cls_msg  
python test_partseg.py --log_dir pointnet2_cls_msg
# Reference By
https://github.com/charlesq34/pointnet2
# Citation
@article{Pytorch_Pointnet_Pointnet2,  
      Author = {Xu Yan},  
      Title = {Pointnet/Pointnet++ Pytorch},  
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},  
      Year = {2019}  
}
