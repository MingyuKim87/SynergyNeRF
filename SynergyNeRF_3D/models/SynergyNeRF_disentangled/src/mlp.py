from typing import Callable, Collection, Dict, Iterable, List, Optional, Sequence, Union

import torch


def positional_encoding(positions, freqs):
    """
    Return positional_encoding results with frequency freqs.
    """
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class General_MLP(torch.nn.Module):
    """
    A general MLP module with potential input including time position encoding(PE): feature PE: fea_pe, 3D position PE: pos_pe,
    view direction PE: view_pe.

    pe > 0: use PE with frequency = pe.
    pe < 0: not use this feautre.
    pe = 0: only use original value.
    """

    def __init__(
        self,
        feat_dim: int,
        output_dim: int,
        fea_pe: int = 6,
        pos_pe: int = 6,
        view_pe: int = 6,
        feature_dim: int = 128,
        n_layers: int = 3,
        use_sigmoid: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.use_fea = fea_pe >= 0
        self.use_pos = pos_pe >= 0
        self.use_view = view_pe >= 0
        self.fea_pe = fea_pe
        self.pos_pe = pos_pe
        self.view_pe = view_pe
        self.use_sigmoid = use_sigmoid

        # Whether use these features as inputs
        if self.use_fea:
            self.feat_dim += 2 * fea_pe * feat_dim
        if self.use_pos:
            self.feat_dim += 3 + 2 * pos_pe * 3
        if self.use_view:
            self.feat_dim += 3 + 2 * view_pe * 3

        assert n_layers >= 2  # Assert at least two layers of MLP
        layers = [torch.nn.Linear(self.feat_dim, feature_dim), torch.nn.ReLU(inplace=True)]

        for _ in range(n_layers - 2):
            layers += [torch.nn.Linear(feature_dim, feature_dim), torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Linear(feature_dim, output_dim)]
        self.mlp = torch.nn.Sequential(*layers)

        if zero_init:
            torch.nn.init.constant_(self.mlp[-1].bias, 0)


    def forward(
        self,
        pts: torch.Tensor,
        viewdirs: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        MLP forward.
        """
        # Collect input data
        indata = [features]
        
        if self.use_fea:
            if self.fea_pe > 0:
                indata += [positional_encoding(features, self.fea_pe)]
        if self.use_pos:
            indata += [pts]
            if self.pos_pe > 0:
                indata += [positional_encoding(pts, self.pos_pe)]
        if self.use_view:
            indata += [viewdirs]
            if self.view_pe > 0:
                indata += [positional_encoding(viewdirs, self.view_pe)]
        mlp_in = torch.cat(indata, dim=-1)

        rgb = self.mlp(mlp_in)
        if self.use_sigmoid:
            rgb = torch.sigmoid(rgb)

        return rgb

class Grid_Coord_Mixing_MLP(torch.nn.Module):
    """
    A general MLP module with potential input including time position encoding(PE): feature PE: fea_pe, 3D position PE: pos_pe,
    view direction PE: view_pe.

    pe > 0: use PE with frequency = pe.
    pe < 0: not use this feautre.
    pe = 0: only use original value.
    """

    def __init__(
        self,
        output_dim: int,
        input_pos_dim: int=3,
        input_feat_dim: int=48,
        pos_pe: int = -1,
        feature_dim: int = 128,
        n_layers: int = 3,
        zero_init: bool = True,
    ):
        super().__init__()

        self.input_pos_dim = input_pos_dim
        self.input_feat_dim = input_feat_dim

        self.use_pos = pos_pe >= 0
        self.pos_pe = pos_pe

        # Set grid feature dimensions
        self.input_dim = input_feat_dim
        
        if self.use_pos:
            self.input_dim += 3 + 2 * pos_pe * 3

        self.hidden_dim = self.input_dim + feature_dim
        
        # layers 1
        layers = [torch.nn.Linear(self.input_dim, feature_dim), torch.nn.ReLU(inplace=True)]

        for _ in range(n_layers - 2):
            layers += [torch.nn.Linear(feature_dim, feature_dim), torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Linear(feature_dim, feature_dim), torch.nn.ReLU(inplace=True)]
        self.mlp_1 = torch.nn.Sequential(*layers)

        # layers 2
        layers = [torch.nn.Linear(self.hidden_dim, feature_dim), torch.nn.ReLU(inplace=True)]
        for _ in range(n_layers - 2):
            layers += [torch.nn.Linear(feature_dim, feature_dim), torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Linear(feature_dim, output_dim)]
        self.mlp_2 = torch.nn.Sequential(*layers)

        # initialize
        relu_gain = torch.nn.init.calculate_gain('relu')
        for layer in self.mlp_1: 
            if type(layer) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight, gain=relu_gain)
                torch.nn.init.zeros_(layer.bias)

        for layer in self.mlp_2: 
            if type(layer) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight, gain=relu_gain)
                torch.nn.init.zeros_(layer.bias)

    def forward(
        self,
        pts: torch.Tensor,
        viewdirs: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        MLP forward.
        """
        # Collect input data
        indata = [features]
        
        if self.use_pos:
            in_pts = [pts]
            if self.pos_pe > 0:
                in_pts += [positional_encoding(pts, self.pos_pe)]

            indata += in_pts
        
        input = torch.cat(indata, dim=-1)
        
        # Feed-forward
        h = self.mlp_1(input)
        input_2 = torch.cat((input, h), dim=-1)
        h = self.mlp_2(input_2)

        # output
        output = h
        
        return output
