from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k=k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, k*k)
        )
    def forward(self, x):
        # 计算变换矩阵并应用
        matrix = self.fc(torch.max(self.conv(x), 2)[0])
        matrix = matrix.view(-1, self.k, self.k)
        return torch.bmm(matrix, x)  # 应用变换

class PointNetfeat(nn.Module):
    '''
        The feature extractor in PointNet, corresponding to the left MLP in the pipeline figure.
        Args:
        d: the dimension of the global feature, default is 1024.
        segmentation: whether to perform segmentation, default is True.
    '''
    def __init__(self, segmentation = False, d=1024):
        super(PointNetfeat, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the feature extractor. ##
        ## ------------------------------------------- ##
        self.seg = segmentation
        self.input_tnet = TNet(k=3)
        self.feature_tnet = TNet(k=64)
        self.d = d
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, d, 1),
            nn.BatchNorm1d(d),
            nn.ReLU(),
        )
        self.mxpool = nn.MaxPool1d(kernel_size=1024) # Need change if change utils.py num_points

    def forward(self, x):
        '''
            If segmentation == True
                return the concatenated global feature and local feature. # (B, d+64, N)
            If segmentation == False
                return the global feature, and the per point feature for cruciality visualization in question b). # (B, d), (B, N, d)
            Here, B is the batch size, N is the number of points, d is the dimension of the global feature.

            Input: B*3*N
            Output: ...
        '''
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        x = self.input_tnet(x)
        x = self.mlp1(x)
        x = self.feature_tnet(x)
        y = x # (B, 64, N)
        x = self.mlp2(x)
        x = self.mlp3(x)
        vis = x
        x = self.mxpool(x)
        if self.seg:
            x = x.view(x.size(0), -1) # (B, d)
            x = x.unsqueeze(2)
            x = x.expand(y.size(0), -1, y.size(2))
            x = torch.cat((y, x), dim=1)
            return x
        else:
            x = x.view(x.size(0), -1)
            vis = vis.transpose(1, 2)
            return x, vis

class PointNetCls1024D(nn.Module):
    '''
        The classifier in PointNet, corresponding to the middle right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k=2):
        super(PointNetCls1024D, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the classifier.        ##
        ## ------------------------------------------- ##
        self.feat = PointNetfeat(segmentation=False, d=1024)
        self.mlp1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.mlp3 = nn.Linear(256, k)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=1024)
            Input: B*N*3
        '''
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        x = x.transpose(2, 1)
        x, _ = self.feat(x)  # (B,1024), (B, N, 1024)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.log_softmax(x)
        x = x.contiguous()
        return x, _


class PointNetCls256D(nn.Module):
    '''
        The classifier in PointNet, corresponding to the upper right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k=2 ):
        super(PointNetCls256D, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the classifier.        ##
        ## ------------------------------------------- ##
        self.feat = PointNetfeat(segmentation=False, d=256)
        self.mlp1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.mlp2 = nn.Linear(128, k)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=256)
        '''
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        x = x.transpose(2, 1)
        x, _ = self.feat(x)  # (B,1024), (B, N, 1024)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.log_softmax(x)
        x = x.contiguous()
        return x, _

class PointNetSeg(nn.Module):
    '''
        The segmentation head in PointNet, corresponding to the lower right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k = 2):
        super(PointNetSeg, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the segmentation head. ##
        ## ------------------------------------------- ##
        self.feat = PointNetfeat(segmentation=True, d=1024)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.mlp4 = nn.Conv1d(128, k, 1)

    def forward(self, x):
        '''
            Input:
                x: the input point cloud. # (B, N, 3)
            Output:
                the log softmax of the segmentation result. # (B, N, k)
        '''
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        x = x.transpose(2, 1)
        x = self.feat(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = F.log_softmax(x, dim=1)
        x = x.transpose(2, 1)
        x = x.contiguous()  # Ensure the output is contiguous in memory
        return x