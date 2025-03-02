import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from argparse import Namespace
from typing import List
from chemprop.nn_utils import get_activation_function, initialize_weights


class DMPNNLayer(nn.Module):
    """Directed Message Passing Neural Network Layer for edge-focused message passing."""

    def __init__(self, hidden_size):
        super(DMPNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W_msg = nn.Linear(hidden_size, hidden_size)
        self.W_update = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, edge_features, edge_index):
        # Message passing step
        message = F.relu(self.W_msg(edge_features))
        aggregated_message = torch.zeros_like(edge_features)
        aggregated_message[edge_index[1]] += message

        # Update step
        updated_edges = self.W_update(aggregated_message, edge_features)
        return updated_edges


class GlobalAttention(nn.Module):
    """Global Attention Mechanism to capture long-range dependencies."""

    def __init__(self, hidden_size, num_heads=4):
        super(GlobalAttention, self).__init__()
        self.attention = MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Apply multi-head attention
        attn_output, _ = self.attention(x, x, x)
        # Residual connection and layer normalization
        x = self.norm(attn_output + x)
        return F.relu(self.linear(x))


class MoleculeModel(nn.Module):
    """DeepTherm with DMPNN, Global Attention, and FFN."""

    def __init__(self, classification: bool, multiclass: bool):
        super(MoleculeModel, self).__init__()
        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace):
        """Creates the message passing encoder for the model."""
        self.encoder = DMPNNLayer(args.hidden_size)

    def create_ffn(self, args: Namespace):
        """Creates the feed-forward network for the model."""
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes

        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * 1
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])
        self.ffn = nn.Sequential(*ffn)

    def forward(self, x, edge_index, additional_features=None):
        """Runs the DeepTherm on input."""
        # DMPNN layer
        edge_features = self.encoder(x)

        # Global attention mechanism
        x = GlobalAttention(edge_features.size(-1))(edge_features)

        # Concatenate additional features if provided
        if additional_features is not None:
            x = torch.cat([x, additional_features], dim=-1)

        # Feed-forward network
        output = self.ffn(x)

        # Classification adjustments
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)

        return output


def build_model(args: Namespace) -> nn.Module:
    """Builds a DeepTherm."""
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification',
                          multiclass=args.dataset_type == 'multiclass')
    model.create_encoder(args)
    model.create_ffn(args)
    initialize_weights(model)
    return model
