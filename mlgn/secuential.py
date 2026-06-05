import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from difflogic import LogicLayer, GroupSum

GATE_NAMES = [
    'FALSE', 'AND', 'A AND NOT B', 'A',
    'NOT A AND B', 'B', 'XOR', 'OR',
    'NOR', 'XNOR', 'NOT B', 'A OR NOT B',
    'NOT A', 'NOT A OR B', 'NAND', 'TRUE',
]

BEST_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'best_model')


class LogicRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        # First layer takes concatenated [input || hidden]
        self.input_layer = LogicLayer(input_dim + hidden_dim, hidden_dim)
        # Additional layers stay in hidden_dim
        self.extra_layers = nn.Sequential(
            *[LogicLayer(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.hidden_dim = hidden_dim

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        h_new = self.input_layer(combined)
        return self.extra_layers(h_new)
        

class SequentialMNISTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=10):
        super().__init__()
        self.cell = LogicRNNCell(input_dim, hidden_dim, num_layers=2)
        self.classifier = GroupSum(k=num_classes, tau=30)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (batch, 1, 28, 28) — feed one row per timestep
        x = x.view(x.shape[0], 28, 28)
        h = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        for t in range(28):
            h = self.cell(x[:, t, :], h)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
LR         = 0.01
NUM_ITERS  = 100_000
EVAL_FREQ  = 2_000
VAL_FRAC   = 0.1
DATA_ROOT  = os.path.join(os.path.dirname(__file__), '..', 'data-mnist')

# ---------------------------------------------------------------------------
# Data
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
model = SequentialMNISTModel(input_dim=28, hidden_dim=16_000, num_classes=10).to(DEVICE)

print(model)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Running on {DEVICE}')

loss_fn   = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cycle(loader):
    while True:
        yield from loader


def evaluate(model, loader):
    was_training = model.training
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x.round()).argmax(-1)  # binarise inputs for discrete gate eval
            correct += (preds == y).sum().item()
            total   += y.size(0)
    model.train(was_training)
    return correct / total


def analyze_gates(model):
    print('\n--- Gate distribution per LogicLayer ---')
    for name, module in model.named_modules():
        if not isinstance(module, LogicLayer):
            continue
        gate_ids = module.weights.argmax(-1).cpu()
        counts = torch.bincount(gate_ids, minlength=16)
        total = counts.sum().item()
        print(f'\n  {name}  ({module.in_dim} -> {module.out_dim})')
        for gate_idx, (gate_name, count) in enumerate(zip(GATE_NAMES, counts.tolist())):
            if count > 0:
                print(f'    [{gate_idx:2d}] {gate_name:<16s}  {count:>6,}  ({100*count/total:.1f}%)')

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
best_val_acc = 0.0
best_state   = None

model.train()
for i, (x, y) in tqdm(enumerate(cycle(train_loader)), total=NUM_ITERS, desc='train'):
    if i >= NUM_ITERS:
        break

    x, y = x.to(DEVICE), y.to(DEVICE)
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
# Final test evaluation on best checkpoint
# ---------------------------------------------------------------------------
print('\n--- Final evaluation (discrete locked gates) ---')
model.load_state_dict(best_state)
test_acc = evaluate(model, test_loader)
print(f'Best val acc : {best_val_acc:.4f}')
print(f'Test acc     : {test_acc:.4f}')

# ---------------------------------------------------------------------------
# Save best model
# ---------------------------------------------------------------------------
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
save_path = os.path.join(BEST_MODEL_DIR, 'model.pt')
torch.save({
    'model_state_dict': best_state,
    'val_acc': best_val_acc,
    'test_acc': test_acc,
}, save_path)
print(f'\nBest model saved to {save_path}')

# ---------------------------------------------------------------------------
# Gate analysis
# ---------------------------------------------------------------------------
model.eval()
analyze_gates(model)




































class SimpleRNN3Layer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # Each cell is one nn.Linear(input + hidden → hidden).
        # This is equivalent to your formula:
        #   h_t = tanh( x_t @ W_ih  +  h_(t-1) @ W_hh  +  b )
        # because cat([x_t, h], dim=1) stacks them into one vector,
        # and one big weight matrix [W_ih | W_hh] applied to that
        # is the same as two separate matmuls added together.

        self.cell1 = nn.Linear(input_size  + hidden_size, hidden_size)  # layer 1: takes raw input
        self.cell2 = nn.Linear(hidden_size + hidden_size, hidden_size)  # layer 2: takes h1 as "input"
        self.cell3 = nn.Linear(hidden_size + hidden_size, hidden_size)  # layer 3: takes h2 as "input"

        self.output = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.shape[0]
        seq_len    = x.shape[1]

        # Each layer keeps its own hidden state across time
        h1 = torch.zeros(batch_size, self.hidden_size)
        h2 = torch.zeros(batch_size, self.hidden_size)
        h3 = torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_len):
            x_t = x[:, t, :]   # (batch, input_size) — one timestep

            # Layer 1: sees the raw input + its own previous hidden state
            # h1_t = tanh( x_t @ W_ih1  +  h1_(t-1) @ W_hh1  +  b1 )
            h1 = torch.tanh(self.cell1(torch.cat([x_t, h1], dim=1)))

            # Layer 2: x_t is now h1 — it never sees the raw pixels directly
            # h2_t = tanh( h1_t @ W_ih2  +  h2_(t-1) @ W_hh2  +  b2 )
            h2 = torch.tanh(self.cell2(torch.cat([h1, h2], dim=1)))

            # Layer 3: same pattern, one level deeper
            # h3_t = tanh( h2_t @ W_ih3  +  h3_(t-1) @ W_hh3  +  b3 )
            h3 = torch.tanh(self.cell3(torch.cat([h2, h3], dim=1)))

        # Only the final hidden state of the deepest layer is used for classification
        return self.output(h3)

# The key insight: cat([x_t, h], dim=1) followed by one Linear is exactly your formula — 
# it just packs W_ih and W_hh side-by-side into one matrix so PyTorch only needs a single matmul. 
# You can think of it as:
# [ x_t | h_(t-1) ]  @  [ W_ih ]  =  x_t @ W_ih  +  h_(t-1) @ W_hh
#                        [ W_hh ]






# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         # Combines input and previous hidden state into new hidden state
#         self.rnn_cell = nn.Linear(input_size + hidden_size, hidden_size)
#         self.output = nn.Linear(hidden_size, output_size)
#         self.hidden_size = hidden_size

#     def forward(self, x):
#         # x shape: (batch, seq_len, input_size)
#         batch_size, seq_len, _ = x.shape

#         # Start with zeros — "no previous context"
#         h = torch.zeros(batch_size, self.hidden_size)

#         for t in range(seq_len):
#             x_t = x[:, t, :]                        # grab timestep t → (batch, input_size)
#             combined = torch.cat([x_t, h], dim=1)   # concat input + prev hidden
#             h = torch.tanh(self.rnn_cell(combined)) # new hidden state

#         return self.output(h)  # use final hidden state for prediction


# Full forward pass (keeping all numbers)

# W = [[ 0.5,  0.3,  0.8, -0.2 ],
#      [ 0.1, -0.6,  0.4,  0.7 ]]

# output weights v = [1.0, 1.0]   (maps h→prediction)
# target = 1.0
# Timestep 1:


# combined_1 = [ 1.0,  2.0,  0.0,  0.0 ]

# z_1 = W @ combined_1
#     = [ 0.5*1.0 + 0.3*2.0 + 0.8*0.0 + (-0.2)*0.0 ]  = [ 1.1  ]
#       [ 0.1*1.0 + (-0.6)*2.0 + 0.4*0.0 + 0.7*0.0  ]  = [ -1.1 ]

# h_1 = tanh(z_1) = [ 0.80, -0.80 ]
# Timestep 2:


# combined_2 = [ 0.5,  0.1,  0.80, -0.80 ]

# z_2 = W @ combined_2
#     = [ 0.5*0.5 + 0.3*0.1 + 0.8*0.80 + (-0.2)*(-0.80) ]  = [ 1.08  ]
#       [ 0.1*0.5 + (-0.6)*0.1 + 0.4*0.80 + 0.7*(-0.80) ]  = [ -0.25 ]

# h_2 = tanh(z_2) = [ 0.79, -0.24 ]
# Prediction and loss:


# pred = v @ h_2 = 1.0*0.79 + 1.0*(-0.24) = 0.55

# loss = (pred - target)^2 = (0.55 - 1.0)^2 = 0.20
# Backward pass — flowing gradient back
# Step 1: gradient of loss w.r.t pred


# d_loss/d_pred = 2 * (pred - target) = 2 * (0.55 - 1.0) = -0.90
# Step 2: gradient reaches h_2


# d_loss/d_h2 = d_loss/d_pred * v = -0.90 * [1.0, 1.0] = [ -0.90, -0.90 ]
# Step 3: gradient through tanh at t=2

# The derivative of tanh is 1 - tanh². We already computed h_2 = tanh(z_2) so:


# tanh_deriv_2 = 1 - h_2^2 = 1 - [0.79², 0.24²] = [0.38, 0.94]

# d_loss/d_z2 = d_loss/d_h2 * tanh_deriv_2
#             = [-0.90, -0.90] * [0.38, 0.94]
#             = [-0.34, -0.85]
# Step 4: W.grad contribution from t=2

# z_2 = W @ combined_2, so the gradient for W is the outer product of the error signal with the input:


# W.grad_t2 = d_loss/d_z2  ⊗  combined_2

#            error signal        what W saw as input
#            ↓                   ↓
# row 0:  -0.34  ×  [ 0.5,  0.1,  0.80, -0.80 ] = [ -0.17, -0.03, -0.27,  0.27 ]
# row 1:  -0.85  ×  [ 0.5,  0.1,  0.80, -0.80 ] = [ -0.43, -0.09, -0.68,  0.68 ]
# Step 5: gradient flows back to h_1 through W


# d_loss/d_combined_2 = W.T @ d_loss/d_z2

#                   = W.T @ [-0.34, -0.85]

# W.T = [[ 0.5,  0.1 ],       row for x part →  ignored (no more timesteps before x_2)
#        [ 0.3, -0.6 ],
#        [ 0.8,  0.4 ],       ← these rows correspond to h_1
#        [-0.2,  0.7 ]]

# d_loss/d_h1 = just the h_1 rows of W.T dotted with the error:
#   [0.8*(-0.34) + 0.4*(-0.85),  (-0.2)*(-0.34) + 0.7*(-0.85)]
# = [-0.27 - 0.34,                 0.07 - 0.60               ]
# = [ -0.61,  -0.53 ]
# Step 6: gradient through tanh at t=1


# tanh_deriv_1 = 1 - h_1^2 = 1 - [0.80², 0.80²] = [0.36, 0.36]

# d_loss/d_z1 = [-0.61, -0.53] * [0.36, 0.36]
#             = [-0.22, -0.19]
# Step 7: W.grad contribution from t=1


# W.grad_t1 = d_loss/d_z1  ⊗  combined_1

# row 0:  -0.22  ×  [ 1.0,  2.0,  0.0,  0.0 ] = [ -0.22, -0.44,  0.0,  0.0 ]
# row 1:  -0.19  ×  [ 1.0,  2.0,  0.0,  0.0 ] = [ -0.19, -0.38,  0.0,  0.0 ]
# Note the last two columns are 0 because h_0 was all zeros — no gradient flows further back.

# Final W.grad = t=2 contribution + t=1 contribution

# W.grad_t2 = [[ -0.17, -0.03, -0.27,  0.27 ],
#              [ -0.43, -0.09, -0.68,  0.68 ]]

# W.grad_t1 = [[ -0.22, -0.44,  0.00,  0.00 ],
#              [ -0.19, -0.38,  0.00,  0.00 ]]

# W.grad     = [[ -0.39, -0.47, -0.27,  0.27 ],
#               [ -0.62, -0.47, -0.68,  0.68 ]]
# Then the weight update:


# W_new = W - lr * W.grad          (e.g. lr = 0.01)

# W_new = [[ 0.5,  0.3,  0.8, -0.2 ]] - 0.01 * [[ -0.39, -0.47, -0.27,  0.27 ]]
#       = [[ 0.504, 0.305, 0.803, -0.203 ]]     ← nudged slightly
