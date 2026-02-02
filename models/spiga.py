import torch
import torch.nn as nn
import torch.nn.functional as F

import models.gnn.pose_proj as pproj
from models.cnn.cnn_multitask import MultitaskCNN
from models.gnn.step_regressor import StepRegressor, RelativePositionEncoder


class SPIGA(nn.Module):
    def __init__(self, num_landmarks=98, num_edges=15, steps=3, **kwargs):
        super(SPIGA, self).__init__()

        # Model parameters
        self.steps = steps          # Cascaded regressors
        self.embedded_dim = 512     # GAT input channel
        self.nstack = 4             # Number of stacked GATs per step
        self.kwindow = 7            # Output cropped window dimension (kernel)
        self.swindow = 0.25         # Scale of the cropped window at first step (Dft. 25% w.r.t the input featuremap)
        self.offset_ratio = [self.swindow/(2**step)/2 for step in range(self.steps)]

        # CNN parameters
        self.num_landmarks = num_landmarks
        self.num_edges = num_edges

        # Initialize backbone
        self.visual_cnn = MultitaskCNN(num_landmarks=self.num_landmarks, num_edges=self.num_edges)
        # Features dimensions
        self.img_res = self.visual_cnn.img_res
        self.visual_res = self.visual_cnn.out_res
        self.visual_dim = self.visual_cnn.ch_dim

        # Initialize Pose head
        self.channels_pose = 6
        self.pose_fc = nn.Linear(self.visual_cnn.ch_dim, self.channels_pose)

        # Initialize feature extractors:
        # Relative positional encoder
        shape_dim = 2 * (self.num_landmarks - 1)
        self.shape_encoder = nn.ModuleList([RelativePositionEncoder(shape_dim, self.embedded_dim, [256, 256]) for _ in range(self.steps)])
        
        # Diagonal mask used to compute relative positions
        diagonal_mask = (torch.ones(self.num_landmarks, self.num_landmarks) - torch.eye(self.num_landmarks)).bool()
        self.diagonal_mask = nn.parameter.Parameter(diagonal_mask, requires_grad=False)

        # Visual feature extractor
        self.theta_S = nn.ParameterList([
            nn.parameter.Parameter(torch.tensor([[self.kwindow / self.visual_res, 0], [0, self.kwindow / self.visual_res]]), requires_grad=False)
            for _ in range(self.steps)
        ])

        self.conv_window = nn.ModuleList([
            nn.Conv2d(self.visual_dim, self.embedded_dim, self.kwindow) for _ in range(self.steps)
        ])

        # Initialize GAT modules
        self.gcn = nn.ModuleList([StepRegressor(self.embedded_dim, 256, self.nstack) for _ in range(self.steps)])

    def forward(self, data):
        # Inputs: Visual features and points projections
        pts_proj, features = self.backbone_forward(data)
        visual_field = features['VisualField'][-1]

        gat_prob = []
        features['Landmarks'] = []
        for step in range(self.steps):
            # Features generation
            embedded_ft = self.extract_embedded(pts_proj, visual_field, step)

            # GAT inference
            offset, gat_prob = self.gcn[step](embedded_ft, gat_prob)
            offset = F.hardtanh(offset)

            # Update coordinates
            pts_proj = pts_proj + self.offset_ratio[step] * offset
            features['Landmarks'].append(pts_proj.clone())

        features['GATProb'] = gat_prob
        return features

    def backbone_forward(self, data):
        print(f"data type: {type(data)}, length: {len(data)}")
        # imgs, model3d, cam_matrix = data
        imgs = data  # 如果 data 只是图像数据
        # 对于模型的其他输入，你可能需要按需要提取
        model3d = None  # 例如，模型3D数据
        cam_matrix = None  # 例如，相机矩阵
        # HourGlass Forward
        features = self.visual_cnn(imgs)

        # Head pose estimation
        pose_raw = features['HGcore'][-1]
        B, L, _, _ = pose_raw.shape
        pose = pose_raw.reshape(B, -1)
        pose = self.pose_fc(pose)
        features['Pose'] = pose.clone()

        # Project model 3D
        euler, trl = pose[:, :3], pose[:, 3:]
        rot = pproj.euler_to_rotation_matrix(euler)
        pts_proj = pproj.projectPoints(model3d, rot, trl, cam_matrix) / self.visual_res

        return pts_proj, features

    def extract_embedded(self, pts_proj, receptive_field, step):
        visual_ft = self.extract_visual_embedded(pts_proj, receptive_field, step)
        shape_ft = self.shape_encoder[step](self.calculate_distances(pts_proj))
        return visual_ft + shape_ft

    def extract_visual_embedded(self, pts_proj, receptive_field, step):
        B, L, _ = pts_proj.shape
        centers = pts_proj + 0.5 / self.visual_res
        theta_trl = (-1 + centers * 2).unsqueeze(-1)

        # 生成 theta_s
        theta_s = self.theta_S[step].unsqueeze(0).repeat(B * L, 1, 1)
        theta = torch.cat((theta_s, theta_trl), -1)  # (B*L, 2, 3)

        # 生成采样 grid
        B, C, _, _ = receptive_field.shape
        grid = F.affine_grid(theta, (B * L, C, self.kwindow, self.kwindow))
        crops = F.grid_sample(receptive_field.expand(B, C, self.visual_res, self.visual_res), grid, padding_mode="border")

        # 调整形状并通过 CNN 提取特征
        crops = crops.view(B * L, C, self.kwindow, self.kwindow)
        visual_ft = self.conv_window[step](crops).view(B, L, -1)

        return visual_ft

    def calculate_distances(self, pts_proj):
        B, L, _ = pts_proj.shape
        pts_a = pts_proj.unsqueeze(2).expand(B, L, L, 2)
        pts_b = pts_a.transpose(1, 2)
        dist = pts_a - pts_b
        return dist[:, self.diagonal_mask, :].reshape(B, L, -1)
