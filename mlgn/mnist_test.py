import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from difflogic import LogicLayer, GroupSum

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE  = 128
LR          = 0.01
NUM_ITERS   = 100_000
EVAL_FREQ   = 2_000
VAL_FRAC    = 0.1        # fraction of train set held out for validation
DATA_ROOT   = os.path.join(os.path.dirname(__file__), '..', 'data-mnist')

# ---------------------------------------------------------------------------
# Data  (train/val split from the 60k training set; test stays untouched)
# ---------------------------------------------------------------------------
transform = torchvision.transforms.ToTensor()

full_train = torchvision.datasets.MNIST(DATA_ROOT, train=True,  download=True, transform=transform)
test_set   = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)

val_size   = int(len(full_train) * VAL_FRAC)
train_size = len(full_train) - val_size
train_set, val_set = torch.utils.data.random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(0)
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# small testing mdoel
# model = torch.nn.Sequential(
#     torch.nn.Flatten(),
#     LogicLayer(784, 16_000),
#     LogicLayer(16_000, 16_000),
#     LogicLayer(16_000, 16_000),
#     LogicLayer(16_000, 16_000),
#     LogicLayer(16_000, 16_000),
#     GroupSum(k=10, tau=30),
# ).to(DEVICE)
# the papers model
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    LogicLayer(784,    64_000),
    LogicLayer(64_000, 64_000),
    LogicLayer(64_000, 64_000),
    LogicLayer(64_000, 64_000),
    LogicLayer(64_000, 64_000),
    LogicLayer(64_000, 64_000),
    GroupSum(k=10, tau=30),
).to(DEVICE)


print(model)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

loss_fn   = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cycle(loader):
    while True:
        yield from loader


def evaluate(model, loader):
    """
    Evaluate in eval mode — this locks each gate to its argmax (discrete logic).
    Inputs are binarized via .round() to match the discrete gate behaviour.
    """
    was_training = model.training
    # Locks network with argmax
    model.eval()
    # locks input to 0 or 1
    with torch.no_grad():
        accs = [
            (model(x.to(DEVICE).round()).argmax(-1) == y.to(DEVICE))
            .float().mean().item()
            for x, y in loader
        ]
    model.train(was_training)
    return float(np.mean(accs))

# ---------------------------------------------------------------------------
# Training loop  (track best val; never touch test_loader until the end)
# ---------------------------------------------------------------------------
best_val_acc   = 0.0
best_state     = None

model.train()
for i, (x, y) in tqdm(enumerate(cycle(train_loader)), total=NUM_ITERS, desc='train'):
    if i >= NUM_ITERS:
        break

    x = x.to(DEVICE)
    y = y.to(DEVICE)

    logits = model(x)
    loss   = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % EVAL_FREQ == 0:
        val_acc = evaluate(model, val_loader)
        print(f'[{i+1:>6}]  loss={loss.item():.4f}  val={val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

# ---------------------------------------------------------------------------
# Final evaluation on test set using the best checkpoint (locked/discrete gates)
# ---------------------------------------------------------------------------
print('\n--- Final evaluation (discrete locked gates) ---')
model.load_state_dict(best_state)
test_acc = evaluate(model, test_loader)
print(f'Best val acc : {best_val_acc:.4f}')
print(f'Test acc     : {test_acc:.4f}')

# --- Final evaluation (discrete locked gates) ---
# Best val acc : 0.9804
# Test acc     : 0.9824