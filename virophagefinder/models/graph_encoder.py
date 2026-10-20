"""Gene organization graph encoder using Graph Attention Network (GAT)."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class GeneGraphEncoder(nn.Module):
    """Encode gene organization structure using Graph Attention Network.

    Constructs a directed graph where:
    - Nodes: predicted genes (features = projected ESM2 protein embeddings)
    - Edges: adjacent genes in genomic order (preserving strand direction)

    Learns synteny patterns and functional associations among genes.
    """

    def __init__(
        self,
        node_feature_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        proj_dim: int = 256,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection (from protein embedding dim to node_feature_dim)
        # This is handled externally; here we assume node features are already projected

        # GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = node_feature_dim if i == 0 else hidden_dim * num_heads
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            heads = num_heads if i < num_layers - 1 else 1

            self.convs.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False,
                )
            )
            self.norms.append(nn.LayerNorm(out_dim * heads if i < num_layers - 1 else out_dim))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

        # Graph-level projection (mean + max pooling concatenated)
        graph_dim = hidden_dim * 2  # mean + max
        self.projection = nn.Sequential(
            nn.Linear(graph_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

    def build_graph(
        self,
        protein_embeddings: torch.Tensor,
        gene_coords: List[Tuple[int, int, str]],
        protein_mask: torch.Tensor,
        device: torch.device,
    ) -> Data:
        """Build a PyG graph from gene information.

        Args:
            protein_embeddings: (num_proteins, node_feature_dim) node features
            gene_coords: List of (start, end, strand) tuples
            protein_mask: (num_proteins,) binary mask for valid proteins
            device: Torch device

        Returns:
            PyG Data object
        """
        valid_idx = torch.where(protein_mask > 0)[0]
        if len(valid_idx) == 0:
            # Empty graph: single dummy node
            x = torch.zeros(1, self.node_feature_dim, device=device)
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            return Data(x=x, edge_index=edge_index)

        x = protein_embeddings[valid_idx]
        n_nodes = len(valid_idx)

        # Build edges: connect adjacent genes (directed by genomic order)
        edges = []
        for i in range(n_nodes - 1):
            # Forward edge (gene i -> gene i+1)
            edges.append([i, i + 1])
            # Also add reverse for bidirectional attention
            edges.append([i + 1, i])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

        return Data(x=x, edge_index=edge_index)

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        gene_coords: List[List[Tuple[int, int, str]]],
        protein_masks: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for a batch of contigs.

        Args:
            protein_embeddings: (batch_size, max_proteins, node_feature_dim)
            gene_coords: Batch of gene coordinate lists
            protein_masks: (batch_size, max_proteins) binary mask
            device: Torch device

        Returns:
            Dictionary with 'graph_embedding' (batch_size, proj_dim)
        """
        batch_size = protein_embeddings.shape[0]
        graphs = []

        for i in range(batch_size):
            graph = self.build_graph(
                protein_embeddings[i],
                gene_coords[i],
                protein_masks[i],
                device,
            )
            graphs.append(graph)

        # Batch graphs
        batch = Batch.from_data_list(graphs).to(device)

        # GAT forward
        x = batch.x
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, batch.edge_index)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)

        # Graph-level pooling
        mean_pool = global_mean_pool(x, batch.batch)
        max_pool = global_max_pool(x, batch.batch)
        graph_embedding = torch.cat([mean_pool, max_pool], dim=-1)

        # Project
        projected = self.projection(graph_embedding)

        return {"graph_embedding": projected, "node_embeddings": x}
