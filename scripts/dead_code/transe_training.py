# general imports
import torch
import pandas as pd
import os

# torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import bernoulli, randint

# kge
from torchkge.data_structures import KnowledgeGraph
from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss

# utils
from tqdm.autonotebook import tqdm
from pathlib import Path
from lightning import seed_everything

# typing
from pandas import DataFrame
from typing import List

# define paths
ROOT: Path = Path(__file__).parent.parent
VISUAL_GENOME: Path = ROOT / "data" / "visual_genome" / "processed"
CSV_FILE: Path = VISUAL_GENOME / "relationships.csv"
SAVE_PATH: Path = ROOT / "models" / "transe_model_vg.pth"

# define some hyper-parameters for training
EMB_DIM: int = 100
LR: float = 0.0004
EPOCHS: int = 1000
B_SIZE: int = 32768
MARGIN: float = 0.5

# seed
SEED: int = 1917

# custom BernoulliNegativeSampler to work with MPS


class BernoulliNegativeSamplerMPS(BernoulliNegativeSampler):
    def corrupt_batch(self, heads, tails, relations, n_neg=None):
        """Changed from kgetorch to work with MPS. Replaced .double()
        calls with .to(torch.float32), because MPS does not support torch.float64.

        For each true triplet, produce a corrupted one assumed to be different
        from any other true triplet. If `heads` and `tails` are cuda objects,
        then the returned tensors are on the GPU.

        Parameters
        ----------
        heads: torch.Tensor, shape: (batch_size)
            Tensor containing the integer key of heads of the relations in the
            current batch.
        tails: torch.Tensor, shape: (batch_size)
            Tensor containing the integer key of tails of the relations in the
            current batch.
        relations: torch.Tensor, shape: (batch_size)
            Tensor containing the integer key of relations in the current
            batch.
        n_neg: int (opt)
            Number of negative sample to create from each fact. It overwrites
            the value set at the construction of the sampler.
        Returns
        -------
        neg_heads: torch.Tensor, shape: (batch_size)
            Tensor containing the integer key of negatively sampled heads of
            the relations in the current batch.
        neg_tails: torch.Tensor, shape: (batch_size)
            Tensor containing the integer key of negatively sampled tails of
            the relations in the current batch.
        """
        if n_neg is None:
            n_neg = self.n_neg

        device = heads.device
        assert (device == tails.device)

        batch_size = heads.shape[0]
        neg_heads = heads.repeat(n_neg)
        neg_tails = tails.repeat(n_neg)

        # Randomly choose which samples will have head/tail corrupted
        self.bern_probs = self.bern_probs.to(device)
        mask = bernoulli(self.bern_probs[relations].repeat(n_neg)).to(torch.float32)
        n_h_cor = int(mask.sum().item())
        neg_heads[mask == 1] = randint(1, self.n_ent,
                                       (n_h_cor,),
                                       device=device)
        neg_tails[mask == 0] = randint(1, self.n_ent,
                                       (batch_size * n_neg - n_h_cor,),
                                       device=device)

        return neg_heads.long(), neg_tails.long()


def main() -> None:
    """
    Code re-elaborated from from kgetorch doc
    """
    # Set random seed for reproducibility
    seed_everything(SEED)
    torch.mps.manual_seed(SEED)

    # load data
    df: DataFrame = pd.read_csv(CSV_FILE, sep="|", header=0)
    kg_train = KnowledgeGraph(
        df=df.astype({"from": str, "to": str, "rel": str})
    )

    # Define the model and criterion
    model = TransEModel(
        EMB_DIM, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
    criterion = MarginLoss(MARGIN)

    # Move everything to MPS if available
    if torch.mps.is_available():
        model.to("mps")
        criterion.to("mps")

    # Define the torch optimizer to be used
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    sampler = BernoulliNegativeSamplerMPS(kg_train)
    dataloader = DataLoader(kg_train, batch_size=B_SIZE)

    iterator = tqdm(range(EPOCHS), unit='epoch')
    for epoch in iterator:
        running_loss = 0.0
        for _, batch in enumerate(dataloader):
            batch: List[torch.Tensor] = [b.to("mps") for b in batch] if torch.mps.is_available() else batch
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)

            optimizer.zero_grad()

            # forward + backward + optimize
            pos, neg = model(h, t, r, n_h, n_t)
            loss: torch.Tensor = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        iterator.set_description(
            'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                                  running_loss / len(dataloader)))

    model.normalize_parameters()

    os.makedirs(SAVE_PATH.parent, exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)


if __name__ == "__main__":
    main()
    print("Training completed.")
