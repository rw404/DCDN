# %%
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm.notebook import tqdm


# %%
class DivisibleDCN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer = nn.Linear(2, 2, False)
        self.projector = nn.Linear(2, 1, False)
    
    def forward(self, x):
        cross_prediction = self.layer(x) / torch.clip(x, min=1.0).float()

        return self.projector(cross_prediction)

# %%
ctrPredictor = DivisibleDCN()

ctrPredictor

# %%
ctrPredictor(torch.tensor([0, 1]).float())

# %%
click_counters = torch.randint(low=0, high=1000, size=(1_000_000,))
show_counters = torch.randint(low=0, high=10000, size=(1_000_000,)) + click_counters

click_counters, show_counters

# %%
optimizer = torch.optim.AdamW(ctrPredictor.parameters(), lr=3e-4, weight_decay=1e-6)

losses = []
for batch_start in tqdm(range(click_counters.size(0) // 100)):
    click_slice = click_counters[batch_start: batch_start+100]
    show_slice = show_counters[batch_start: batch_start+100]

    input_batch = torch.stack([click_slice, show_slice], dim=-1)

    optimizer.zero_grad()
    predicted_ctrs = ctrPredictor(input_batch.float())
    true_ctrs = click_slice.float() / show_slice.float()
    loss = torch.nn.functional.mse_loss(true_ctrs.unsqueeze(-1), predicted_ctrs)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())


# %%
plt.figure(figsize=(10, 5))
plt.title("Ctr mse losses", fontsize=20)
plt.plot(range(click_counters.size(0) // 100), losses, label="losses")
plt.grid(True)
plt.legend(fontsize=12)
plt.xlabel("Step")
plt.ylabel("MSE loss")
plt.show()

# %% [markdown]
# # Test

# %%
click_counters_test = torch.randint(low=0, high=1000, size=(10,))
show_counters_test = torch.randint(low=0, high=10000, size=(10,)) + click_counters_test

click_counters_test, show_counters_test

# %%
truth_ctrs_test = click_counters_test / show_counters_test

ctrPredictor.eval()
with torch.no_grad():
    predicted_ctrs_test = ctrPredictor(torch.stack([click_counters_test, show_counters_test], dim=1).float())

print(f"True ctrs test: {truth_ctrs_test}")
print(f"Pred ctrs test: {predicted_ctrs_test}")

# %%



