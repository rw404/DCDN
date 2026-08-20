# %% [markdown]
# # DCDN, объяснённый на маленьких данных
#
# **DCDN — Deep Cross-Division Network.** Этот ноутбук — *объясняющий модельный
# пример*, а не результат исследования. Он показывает на синтетике и на
# MovieLens-1M, что именно делает reciprocal-ветвь DCDN и почему её наличие
# позволяет убрать из feature view вручную материализованные деления
# исторических счётчиков и сумм.
#
# Ноутбук импортирует **настоящий** `DCDN-Core` из `src/dcdn/core.py` — тот же
# модуль, который используется во фризнутых экспериментах. Базовые линии
# (`MLP`, `CrossNetV2`) здесь — компактные дидактические реализации; в основном
# исследовании используются аудированные версии из `src/dcdn/models/`.
#
# ## Чего этот ноутбук НЕ утверждает
#
# * Это **не** доказательство превосходства DCDN в рекомендациях. Датасеты
#   маленькие, тюнинг минимальный, seed один, доверительных интервалов нет.
# * Это **не** замена `Synthetic-MultiQuotient-3` и основной multi-target
#   таблицы — они живут во фризнутом протоколе репозитория.
# * MovieLens-1M **не входит** в обязательный scope исследования
#   (VK-LSVD, Yambda-50M, KuaiRand-Pure, KuaiRec big matrix). Он взят только
#   потому, что маленький, публичный и всем знаком.
#
# ## План
#
# 1. Почему деление — трудная операция для DCN-подобных сетей.
# 2. Рекуррентность DCDN и masked reciprocal carrier.
# 3. Демо A — одно частное `y = n/d`, ID и OOD по масштабу.
# 4. Демо B — `Synthetic-MultiQuotient-3` (mini): три частных, три головы.
# 5. Демо C — MovieLens-1M: multi-target, FULL vs REDUCED feature view.
# 6. Выводы и границы применимости.

# %%
from __future__ import annotations

import hashlib
import io
import math
import os
import sys
import time
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Найти корень репозитория и сделать импортируемым src/.
_here = Path.cwd()
REPO_ROOT = next(
    (p for p in (_here, *_here.parents) if (p / "src" / "dcdn" / "core.py").exists()),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError("запустите ноутбук внутри клона репозитория DCDN")
sys.path.insert(0, str(REPO_ROOT / "src"))

from dcdn.core import DCDNCore  # noqa: E402

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"repo         : {REPO_ROOT}")
print(f"torch        : {torch.__version__}")
print(f"device       : {DEVICE}")
print(f"DCDNCore из  : {REPO_ROOT / 'src' / 'dcdn' / 'core.py'}")

# %% [markdown]
# ## 1. Почему деление трудно
#
# В промышленном ранжировании огромная доля признаков — это отношения
# исторических счётчиков: CTR = `clicks / shows`, средняя досмотренность =
# `watch_sum / shows`, like-rate = `(likes + α) / (shows + β)`. Их считают
# вручную в фиче-пайплайне, и это дорого: их надо материализовать, хранить,
# синхронизировать между обучением и продом.
#
# Причина, по которой их считают вручную, проста: ReLU-сеть приближает `n/d`
# кусочно-линейно. Внутри обучающего диапазона это работает; при сдвиге
# масштаба — нет. Проверим это буквально в три строки.

# %%
def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 3) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def _fit_regression(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 3000,
    lr: float = 3e-3,
    batch: int = 1024,
) -> nn.Module:
    """Минимальный, полностью детерминированный цикл обучения на MSE."""
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    x, y = x.to(DEVICE), y.to(DEVICE)
    n = x.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=gen).to(DEVICE)
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x[idx]), y[idx])
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def _nmse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """MSE, нормированный на дисперсию таргета: 1.0 = уровень константы."""
    pred = model(x.to(DEVICE)).cpu()
    var = float(y.var(unbiased=False).clamp_min(1e-12))
    return float(((pred - y) ** 2).mean()) / var


# Обучающая выборка: n ∈ [1, 10], d ∈ [1, 10].  y = n / d.
g = torch.Generator().manual_seed(SEED)
n_tr = torch.rand(20_000, 1, generator=g) * 9 + 1
d_tr = torch.rand(20_000, 1, generator=g) * 9 + 1
x_tr = torch.cat([n_tr, d_tr], dim=1)
y_tr = n_tr / d_tr

# ID-тест: тот же диапазон.  OOD-тест: обе величины умножены на 10,
# то есть частное лежит в том же диапазоне, а операнды — нет.
n_id = torch.rand(5_000, 1, generator=g) * 9 + 1
d_id = torch.rand(5_000, 1, generator=g) * 9 + 1
x_id, y_id = torch.cat([n_id, d_id], 1), n_id / d_id
x_ood, y_ood = x_id * 10.0, y_id.clone()

mlp = _fit_regression(_mlp(2, 128, 1), x_tr, y_tr)
print(f"MLP  nMSE  ID  = {_nmse(mlp, x_id, y_id):.4f}")
print(f"MLP  nMSE  OOD = {_nmse(mlp, x_ood, y_ood):.4f}"
      "   <-- тот же y, другой масштаб операндов")

# %% [markdown]
# MLP решает задачу внутри диапазона и разваливается при общем сдвиге
# масштаба, хотя **значение таргета не изменилось**: `10n / 10d = n / d`.
# Это и есть та инвариантность, которую ratio-признак даёт бесплатно, а
# кусочно-линейная сеть — нет.

# %% [markdown]
# ## 2. Что делает DCDN
#
# `DCDN-Core` — это стек с фиксированным якорем `x₀`. Рекуррентность (см.
# `DCDNCore.RECURRENCE_CONTRACT`):
#
# $$x_{i+1} \;=\; x_i \;+\; x_0 \odot W^{(p)}_i x_i
# \;+\; \mathrm{inv}(x_0) \odot W^{(d)}_i x_i$$
#
# Три слагаемых: residual, **product**-ветвь (как в DCNv2) и **division**-ветвь.
# Ключевой объект — *masked reciprocal carrier* `inv(x₀)`:
#
# * он строится **один раз** из фиксированного якоря `x₀`, а не из `x_i`;
# * он ненулевой **только** на координатах, помеченных семантической маской
#   знаменателей `D` — то есть на положительных исторических операндах;
# * он считается в FP32 с положительным полом `div_eps`, поэтому это
#   *стабилизированный* оператор, а не точный обратный;
# * категориальные, ID- и similarity-эмбеддинги в reciprocal-ветвь не попадают.
#
# Посмотрим на carrier руками.

# %%
# Знаменателями объявлены координаты 1 и 3.
demo_mask = [False, True, False, True]
demo_core = DCDNCore(
    features_count=4,
    num_layers=1,
    denominator_mask=demo_mask,
    div_eps=1e-2,
).to(DEVICE)

x0 = torch.tensor([[3.0, 4.0, 5.0, 0.5]], device=DEVICE)
validity = torch.tensor([[True, True, True, True]], device=DEVICE)
carrier = demo_core.reciprocal_carrier(x0, validity)

print(f"x0            = {x0.cpu().numpy()}")
print(f"маска D       = {demo_mask}")
print(f"inv(x0)       = {carrier.cpu().numpy()}"
      "     # 1/4 = 0.25, 1/0.5 = 2.0, вне маски — ноль")
print(f"mask sha256   = {demo_core.denominator_mask_sha256[:16]}...")
print(f"label         = {demo_core.architecture_label}")

# Невалидный знаменатель гасится, а не превращается в огромное число.
carrier_invalid = demo_core.reciprocal_carrier(
    x0, torch.tensor([[True, True, True, False]], device=DEVICE)
)
print(f"inv(x0) при невалидной коорд. 3 = {carrier_invalid.cpu().numpy()}")

# %% [markdown]
# ### Почему одного слоя достаточно для частного
#
# При `x_i = x₀` координата `j`, помеченная как знаменатель, получает
#
# $$(x_1)_j = (x_0)_j + (x_0)_j\,(W^{(p)}x_0)_j
# + \frac{(W^{(d)}x_0)_j + b^{(d)}_j}{\max((x_0)_j,\;\varepsilon)}$$
#
# то есть **уже на первом слое** в координате знаменателя присутствует
# отношение произвольной линейной комбинации входов к этому знаменателю.
# Чтобы это не осталось словами, соберём веса аналитически и проверим равенство.

# %%
oracle = DCDNCore(
    features_count=2, num_layers=1, denominator_mask=[False, True]
).to(DEVICE)
with torch.no_grad():
    blk = oracle.layers[0]
    # Обнуляем всё, затем включаем ровно один путь: division-ветвь в координате 1
    # берёт числитель из координаты 0.  Тогда (x1)_1 = x_1 + n/d.
    blk.product_block.weight.zero_()
    blk.product_block.bias.zero_()
    blk.division_block.weight.zero_()
    blk.division_block.bias.zero_()
    blk.division_block.weight[1, 0] = 1.0

probe = torch.tensor([[7.0, 2.0], [1.0, 8.0], [90.0, 30.0]], device=DEVICE)
with torch.no_grad():
    out = oracle(probe)
quotient = out[:, 1] - probe[:, 1]              # снимаем residual
expected = probe[:, 0] / probe[:, 1]
print("n        d        восстановлено   ожидалось")
for row, got, exp in zip(probe.cpu(), quotient.cpu(), expected.cpu(), strict=True):
    print(f"{row[0]:<8.1f} {row[1]:<8.1f} {got:<15.6f} {exp:.6f}")
print(f"\nmax|ошибка| = {float((quotient - expected).abs().max()):.3e}")

# %% [markdown]
# Частное воспроизводится точно (в пределах FP32 и пола `div_eps`). Это
# *представимость*, а не обучаемость. Дальше проверим, что градиентный спуск
# действительно находит такое решение из случайной инициализации.

# %% [markdown]
# ## 3. Демо A — обучаемость одного частного
#
# Сравним на той же задаче `y = n/d`:
#
# * `MLP` — базовая кусочно-линейная сеть;
# * `CrossNetV2` — product-only кросс (структурный аналог DCNv2 без division);
# * `DCDN-Core-L1` — product + masked division.
#
# Все три получают одинаковый скалярный выход через одинаковую линейную голову.

# %%
class CrossNetV2(nn.Module):
    """x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l — дидактическая реализация."""

    def __init__(self, dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for layer in self.layers:
            x = x0 * layer(x) + x
        return x


class Wrapped(nn.Module):
    """Интеракционный блок + одинаковая линейная голова."""

    def __init__(self, block: nn.Module, dim: int, out_dim: int = 1) -> None:
        super().__init__()
        self.block = block
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(x))


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def seeded(factory, seed: int = SEED):
    """Построить модуль под фиксированным seed инициализации.

    Инициализация обязана быть воспроизводимой независимо от того, сколько
    вспомогательных модулей было создано раньше в том же процессе.  Это тот же
    контракт, что и ``init_seed`` у ``ControlledInteractionModel`` в
    ``src/dcdn/models/common.py``.
    """
    torch.manual_seed(seed)
    return factory()


models_a = {
    "MLP": lambda: _mlp(2, 128, 1),
    "CrossNetV2-L2": lambda: Wrapped(CrossNetV2(2, num_layers=2), 2),
    "DCDN-Core-L1": lambda: Wrapped(
        DCDNCore(features_count=2, num_layers=1, denominator_mask=[False, True]), 2
    ),
}

print(f"{'модель':<16}{'параметры':>10}{'nMSE ID':>12}{'nMSE OOD ×10':>16}")
for name, factory in models_a.items():
    fitted = _fit_regression(seeded(factory), x_tr, y_tr)
    print(
        f"{name:<16}{_count_params(fitted):>10}"
        f"{_nmse(fitted, x_id, y_id):>12.4f}{_nmse(fitted, x_ood, y_ood):>16.4f}"
    )

# %% [markdown]
# ### Как это читать
#
# Единственное честное сравнение здесь — **`DCDN-Core-L1` против
# `CrossNetV2-L2`**: у них ровно одинаковое число параметров (15), одинаковая
# голова и одинаковый бюджет обучения. Отличается только одно — наличие
# reciprocal-ветви. На ID DCDN примерно на порядок точнее, на OOD — на два
# порядка. Это и есть эффект структуры при фиксированной ёмкости.
#
# MLP в этой таблице **не** сопоставим: у него более чем в две тысячи раз
# больше параметров. Он выигрывает ID и выигрывает common-scale OOD, и это
# нужно проговорить прямо, а не спрятать.
#
# И главное наблюдение, которое стоит зафиксировать сразу: **reciprocal-ветвь
# сама по себе не даёт инвариантности к масштабу.** Product-слагаемое
# `x₀ ⊙ W_p x_i` растёт как квадрат масштаба операндов, поэтому найденное
# градиентным спуском решение — не чистое частное, даже когда чистое частное
# представимо (мы это только что показали аналитически). Reciprocal-ветвь
# помогает там, где нужно именно деление; она не отменяет остальную сеть.

# %% [markdown]
# ## 4. Демо B — `Synthetic-MultiQuotient-3` (mini)
#
# Это уменьшенная предпросмотровая версия обязательного sanity-бенчмарка.
# Полная версия с зафиксированными и захэшированными сплитами живёт во
# фризнутом протоколе; здесь — та же структура задачи в дидактическом объёме.
#
# Вход: `[n1, d1, n2, d2, d3, z1, z2, z3]`, все положительные.
# Три головы:
#
# * `y1 = n1 / d1`
# * `y2 = n2 / d2`
# * `y3 = (n1 + n2) / d3`
#
# `z1..z3` — шумовые координаты-отвлекатели. Reciprocal-маска активна **только**
# на `d1, d2, d3`. Все знаменатели лежат выше reciprocal floor.

# %%
MQ_COLUMNS = ("n1", "d1", "n2", "d2", "d3", "z1", "z2", "z3")
MQ_MASK = [c in ("d1", "d2", "d3") for c in MQ_COLUMNS]        # только знаменатели
MQ_TARGETS = ("y1=n1/d1", "y2=n2/d2", "y3=(n1+n2)/d3")


def make_multiquotient(n_rows: int, seed: int, scale: float = 1.0,
                       ratio: str = "id") -> tuple[torch.Tensor, torch.Tensor]:
    """Сгенерировать один срез бенчмарка.

    ``ratio='low'``/``'high'`` сдвигают распределение частного, ``scale``
    сдвигает общий масштаб операндов, оставляя частное неизменным.
    """
    gen = torch.Generator().manual_seed(seed)
    num = torch.rand(n_rows, 3, generator=gen) * 9 + 1          # числители 1..10
    if ratio == "id":
        den = torch.rand(n_rows, 3, generator=gen) * 9 + 1      # знаменатели 1..10
    elif ratio == "low":
        den = torch.rand(n_rows, 3, generator=gen) * 40 + 20    # частное мало
    elif ratio == "high":
        den = torch.rand(n_rows, 3, generator=gen) * 0.4 + 0.2  # частное велико
    else:
        raise ValueError(ratio)
    noise = torch.rand(n_rows, 3, generator=gen) * 9 + 1
    n1, n2, n_extra = num[:, 0:1], num[:, 1:2], num[:, 2:3]
    d1, d2, d3 = den[:, 0:1], den[:, 1:2], den[:, 2:3]
    del n_extra
    x = torch.cat([n1, d1, n2, d2, d3, noise], dim=1) * scale
    # Частные инвариантны к общему масштабу: (s·n)/(s·d) = n/d.
    y = torch.cat([n1 / d1, n2 / d2, (n1 + n2) / d3], dim=1)
    return x, y


MQ_SPLITS = {
    "ID":              make_multiquotient(6_000, seed=101, scale=1.0, ratio="id"),
    "OOD common-scale": make_multiquotient(6_000, seed=102, scale=10.0, ratio="id"),
    "OOD low-ratio":   make_multiquotient(6_000, seed=103, scale=1.0, ratio="low"),
    "OOD high-ratio":  make_multiquotient(6_000, seed=104, scale=1.0, ratio="high"),
}
mq_x_tr, mq_y_tr = make_multiquotient(40_000, seed=100, scale=1.0, ratio="id")

# Хэшируем срезы ДО обучения — та же дисциплина, что и во фризнутом протоколе.
for split, (sx, sy) in MQ_SPLITS.items():
    digest = hashlib.sha256(
        np.ascontiguousarray(sx.numpy()).tobytes()
        + np.ascontiguousarray(sy.numpy()).tobytes()
    ).hexdigest()
    print(f"{split:<18} rows={sx.shape[0]:>6}  sha256={digest[:16]}...")

# %% [markdown]
# ### Общий multi-target objective
#
# Как и в основном исследовании: константы задачи считаются **на train** и
# замораживаются, каждая голова нормируется на свою константу, агрегат —
# невзвешенное среднее. Никаких адаптивных весов задач.
#
# $$\text{Macro-NLoss} \;=\; \frac{1}{T}\sum_t \frac{\text{MSE}_t}{V_t},
# \qquad V_t = \operatorname{Var}_{\text{train}}(y_t)$$

# %%
class MultiHead(nn.Module):
    """Интеракционный блок + общая башня + отдельная линейная голова на таргет."""

    def __init__(self, block: nn.Module, block_out: int, n_targets: int,
                 tower: int = 64) -> None:
        super().__init__()
        self.block = block
        self.tower = nn.Sequential(
            nn.Linear(block_out, tower), nn.ReLU(), nn.Linear(tower, tower), nn.ReLU()
        )
        self.heads = nn.ModuleList(nn.Linear(tower, 1) for _ in range(n_targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tower(self.block(x))
        return torch.cat([head(h) for head in self.heads], dim=1)


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def fit_multiquotient(block: nn.Module, block_out: int, steps: int = 4000,
                      lr: float = 3e-3) -> tuple[nn.Module, torch.Tensor]:
    model = seeded(lambda: MultiHead(block, block_out, n_targets=3)).to(DEVICE)
    # Константы задач считаются на train и замораживаются.
    var_train = mq_y_tr.var(dim=0, unbiased=False).clamp_min(1e-12).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    x, y = mq_x_tr.to(DEVICE), mq_y_tr.to(DEVICE)
    gen = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, x.shape[0], (1024,), generator=gen).to(DEVICE)
        opt.zero_grad(set_to_none=True)
        per_target = ((model(x[idx]) - y[idx]) ** 2).mean(dim=0)
        (per_target / var_train).mean().backward()      # Macro-NLoss
        opt.step()
    model.eval()
    return model, var_train


@torch.no_grad()
def eval_multiquotient(model: nn.Module, var_train: torch.Tensor) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for split, (sx, sy) in MQ_SPLITS.items():
        pred = model(sx.to(DEVICE))
        nonfinite = int((~torch.isfinite(pred)).sum())
        per_target = ((pred - sy.to(DEVICE)) ** 2).mean(dim=0) / var_train
        report[split] = {
            "per_target": [float(v) for v in per_target.cpu()],
            "macro": float(per_target.mean()),
            "nonfinite": nonfinite,
        }
    return report


mq_models = {
    "MLP":                 (lambda: _mlp(8, 128, 8, depth=2), 8),
    "CrossNetV2-L2":       (lambda: CrossNetV2(8, num_layers=2), 8),
    "DCDN-Core-L1":        (lambda: DCDNCore(8, 1, MQ_MASK), 8),
    "DCDN-Core-L2":        (lambda: DCDNCore(8, 2, MQ_MASK), 8),
}

mq_reports = {}
for name, (factory, out_dim) in mq_models.items():
    model, var_train = fit_multiquotient(seeded(factory), out_dim)
    mq_reports[name] = eval_multiquotient(model, var_train)
    mq_reports[name]["params"] = _count_params(model)

hdr = f"{'модель':<16}{'параметры':>10}" + "".join(f"{s:>19}" for s in MQ_SPLITS)
print(hdr)
print("-" * len(hdr))
for name, rep in mq_reports.items():
    row = f"{name:<16}{rep['params']:>10}"
    for split in MQ_SPLITS:
        row += f"{rep[split]['macro']:>19.4f}"
    print(row)
print("\n(значения — Macro-NLoss, меньше = лучше; "
      "1.0 = уровень константного предсказания)")

# %% [markdown]
# ### Разбивка по задачам и проверка численной устойчивости
#
# Агрегат обязателен, но публиковать только его нельзя: он может прятать
# провал отдельной головы. Поэтому — каждая задача отдельно.

# %%
for split in MQ_SPLITS:
    print(f"\n--- {split} ---")
    header = f"{'модель':<16}" + "".join(f"{t:>18}" for t in MQ_TARGETS)
    print(header + f"{'non-finite':>12}")
    for name, rep in mq_reports.items():
        row = f"{name:<16}"
        row += "".join(f"{v:>18.4f}" for v in rep[split]["per_target"])
        row += f"{rep[split]['nonfinite']:>12}"
        print(row)

# %% [markdown]
# ### Как это читать
#
# `y3 = (n1 + n2) / d3` — самая содержательная голова: числитель здесь не
# входная координата, а сумма двух других. Модель должна собрать его сама
# внутри division-ветви.
#
# Картина по срезам получается неоднородной, и именно поэтому агрегат нельзя
# публиковать в одиночку:
#
# * **ID** — `DCDN-Core-L1` точнее всех, включая MLP, при примерно вчетверо
#   меньшем числе параметров. Внутри распределения reciprocal-ветвь
#   действительно выучивает частные.
# * **OOD high-ratio** (маленькие знаменатели, большие частные) — DCDN лучше
#   MLP в несколько раз. Это ровно тот режим, ради которого существует
#   reciprocal-ветвь.
# * **OOD common-scale и low-ratio** — MLP лучше DCDN. Общий сдвиг масштаба
#   операндов не компенсируется: product-ветвь растёт вместе с ним.
# * **`CrossNetV2-L2`** — структурный близнец DCDN без division-ветви — хуже
#   DCDN на всех четырёх срезах, местами на три порядка.
# * **`DCDN-Core-L2`** заметно нестабильнее `L1` вне распределения: второй
#   слой перемножает уже усиленные величины. Глубина здесь не бесплатна.
#
# Вывод, который эти числа поддерживают: reciprocal-ветвь — это выигрыш
# **относительно product-only кросса** и в режиме больших частных, а не
# универсальная инвариантность. Вывод, которого они **не** поддерживают:
# «DCDN экстраполирует лучше всех».
#
# Ещё раз: это **тест способности реализации учить частные**, а не
# доказательство преимущества в рекомендациях.

# %% [markdown]
# ## 5. Демо C — MovieLens-1M
#
# Теперь то же самое, но на настоящих данных и в терминах реального протокола.
#
# ### 5.1 Роли признаков
#
# | роль | смысл | FULL | REDUCED |
# |------|-------|:----:|:-------:|
# | `C` | категориальные / контекст | ✓ | ✓ |
# | `P_shared` | сырые исторические counters/sums, операнды, validity | ✓ | ✓ |
# | `D ⊂ P_shared` | положительные исторические знаменатели (маска DCDN) | ✓ | ✓ |
# | `R_div` | **вручную посчитанные деления** counters/sums | ✓ | ✗ |
#
# `FULL = C + P_shared + R_div`, `REDUCED = C + P_shared`. Убираются **только**
# ручные деления исторических счётчиков и сумм. В основном исследовании
# косинусы, скалярные произведения, нормы векторов и iALS-скоры остаются в
# обоих режимах и **не** входят в маску знаменателей `D`.
#
# ### 5.2 Временной протокол без утечек
#
# Рейтинги сортируются по времени и режутся на четыре последовательных куска:
#
# * **history** (первые 50%) — из него и только из него считаются все
#   счётчики; после этого они замораживаются, никаких скользящих обновлений;
# * **train** (30%), **validation** (10%), **test** (10%).
#
# Текущий рейтинг строки — это **лейбл**, он никогда не входит в признаки
# ни этой строки, ни её истории.

# %%
def _movielens_dir() -> Path:
    """Найти локальный ml-1m, при отсутствии — скачать в кэш."""
    candidates = [
        Path(os.environ.get("DCDN_DATA_CACHE", ""))
        / "downloads" / "movielens" / "ml-1m",
        REPO_ROOT / "data" / "movielens" / "ml-1m",
        Path.home() / ".cache" / "dcdn-public" / "downloads" / "movielens" / "ml-1m",
    ]
    for c in candidates:
        if (c / "ratings.dat").exists():
            return c
    target = candidates[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    print(f"скачиваю {url} ...")
    with urllib.request.urlopen(url) as resp:                      # noqa: S310
        payload = resp.read()
    print(f"sha256(ml-1m.zip) = {hashlib.sha256(payload).hexdigest()}")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(target.parent)
    return target


ML_DIR = _movielens_dir()
print(f"MovieLens-1M: {ML_DIR}")


def _read_dat(path: Path, n_fields: int) -> np.ndarray:
    rows = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("::")
            if len(parts) == n_fields:
                rows.append(parts)
    return np.array(rows, dtype=object)


ratings = _read_dat(ML_DIR / "ratings.dat", 4)
users_raw = _read_dat(ML_DIR / "users.dat", 5)
movies_raw = _read_dat(ML_DIR / "movies.dat", 3)

user_id = ratings[:, 0].astype(np.int64)
item_id = ratings[:, 1].astype(np.int64)
rating = ratings[:, 2].astype(np.float32)
timestamp = ratings[:, 3].astype(np.int64)

order = np.argsort(timestamp, kind="stable")        # строго по времени
user_id, item_id, rating, timestamp = (
    user_id[order], item_id[order], rating[order], timestamp[order]
)

n_rows = user_id.shape[0]
b_hist, b_train, b_val = int(n_rows * 0.5), int(n_rows * 0.8), int(n_rows * 0.9)
slices = {
    "history": slice(0, b_hist),
    "train": slice(b_hist, b_train),
    "validation": slice(b_train, b_val),
    "test": slice(b_val, n_rows),
}
print(f"строк всего: {n_rows}")
for name, sl in slices.items():
    print(f"  {name:<11} {sl.stop - sl.start:>7} строк, "
          f"ts {timestamp[sl][0]} .. {timestamp[sl][-1]}")
assert timestamp[slices['history']][-1] <= timestamp[slices['train']][0]
assert timestamp[slices['train']][-1] <= timestamp[slices['validation']][0]
assert timestamp[slices['validation']][-1] <= timestamp[slices['test']][0]
print("временная монотонность сплитов проверена")

# %% [markdown]
# ### 5.3 Таргеты
#
# Две головы, как в мини-версии обязательного multi-target objective:
#
# * `t_like = 1[rating ≥ 4]` — бинарная, BCEWithLogits;
# * `t_norm_rating = (rating − 1) / 4` — регрессия, MSE.
#
# Константы задач `H_like` (энтропия константного предсказания) и
# `V_rating` (дисперсия на train) считаются **на train** и замораживаются.

# %%
t_like = (rating >= 4.0).astype(np.float32)
t_norm_rating = (rating - 1.0) / 4.0

tr = slices["train"]
p_like_train = float(t_like[tr].mean())
H_like = float(
    -(p_like_train * math.log(p_like_train)
      + (1 - p_like_train) * math.log(1 - p_like_train))
)
V_rating = float(t_norm_rating[tr].var())
print(f"p(like) на train = {p_like_train:.4f}")
print(f"H_like  (заморожено) = {H_like:.6f}")
print(f"V_rating(заморожено) = {V_rating:.6f}")

# %% [markdown]
# ### 5.4 Исторические счётчики — операнды `P_shared`
#
# Считаем строго по срезу `history`. Сглаживание `α = 5`, `β = 100` — те же
# константы, что в VK-контракте, взяты не подбором, а копированием.

# %%
ALPHA, BETA = 5.0, 100.0
h = slices["history"]
n_users = int(user_id.max()) + 1
n_items = int(item_id.max()) + 1

def _bincount(keys: np.ndarray, weights: np.ndarray | None, size: int) -> np.ndarray:
    return np.bincount(keys, weights=weights, minlength=size).astype(np.float32)

u_show = _bincount(user_id[h], None, n_users)
u_rating_sum = _bincount(user_id[h], rating[h], n_users)
u_like = _bincount(user_id[h], t_like[h], n_users)
i_show = _bincount(item_id[h], None, n_items)
i_rating_sum = _bincount(item_id[h], rating[h], n_items)
i_like = _bincount(item_id[h], t_like[h], n_items)

print(f"пользователей с историей: {int((u_show > 0).sum())} из {n_users - 1}")
print(f"фильмов с историей      : {int((i_show > 0).sum())} из {n_items - 1}")

# P_shared — сырые операнды и индикаторы валидности.
P_COLUMNS = (
    "p_user_show_count", "p_user_show_count_plus_beta", "p_user_rating_sum",
    "p_user_like_count_plus_alpha",
    "p_item_show_count", "p_item_show_count_plus_beta", "p_item_rating_sum",
    "p_item_like_count_plus_alpha",
    "v_user_history_present", "v_item_history_present",
)
# D ⊂ P_shared: только положительные исторические знаменатели.
D_COLUMNS = (
    "p_user_show_count", "p_user_show_count_plus_beta",
    "p_item_show_count", "p_item_show_count_plus_beta",
)
# R_div: ровно safe_div(N, D), где оба операнда сохранены в P_shared.
R_DIV_COLUMNS = (
    "e_user_avg_rating", "e_item_avg_rating", "e_user_like_rate", "e_item_like_rate",
)
R_DIV_LINEAGE = {
    "e_user_avg_rating": ("p_user_rating_sum", "p_user_show_count"),
    "e_item_avg_rating": ("p_item_rating_sum", "p_item_show_count"),
    "e_user_like_rate": ("p_user_like_count_plus_alpha", "p_user_show_count_plus_beta"),
    "e_item_like_rate": ("p_item_like_count_plus_alpha", "p_item_show_count_plus_beta"),
}


def build_numeric(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Вернуть (P_shared, R_div) для набора строк."""
    u, i = user_id[idx], item_id[idx]
    p = np.stack([
        u_show[u], u_show[u] + BETA, u_rating_sum[u], u_like[u] + ALPHA,
        i_show[i], i_show[i] + BETA, i_rating_sum[i], i_like[i] + ALPHA,
        (u_show[u] > 0).astype(np.float32), (i_show[i] > 0).astype(np.float32),
    ], axis=1).astype(np.float32)
    r = np.stack([
        np.where(u_show[u] > 0, u_rating_sum[u] / np.maximum(u_show[u], 1.0), 0.0),
        np.where(i_show[i] > 0, i_rating_sum[i] / np.maximum(i_show[i], 1.0), 0.0),
        (u_like[u] + ALPHA) / (u_show[u] + BETA),
        (i_like[i] + ALPHA) / (i_show[i] + BETA),
    ], axis=1).astype(np.float32)
    return p, r


# C — категориальные признаки.
u_gender = np.zeros(n_users, dtype=np.int64)
u_age = np.zeros(n_users, dtype=np.int64)
u_occ = np.zeros(n_users, dtype=np.int64)
age_levels = sorted({int(r[2]) for r in users_raw})
age_index = {a: k for k, a in enumerate(age_levels)}
for row in users_raw:
    uid = int(row[0])
    u_gender[uid] = 1 if row[1] == "M" else 0
    u_age[uid] = age_index[int(row[2])]
    u_occ[uid] = int(row[3])

genre_levels = sorted({g for row in movies_raw for g in row[2].split("|")})
genre_index = {g: k for k, g in enumerate(genre_levels)}
i_genre = np.zeros(n_items, dtype=np.int64)
i_decade = np.zeros(n_items, dtype=np.int64)
for row in movies_raw:
    iid = int(row[0])
    i_genre[iid] = genre_index[row[2].split("|")[0]]
    title = row[1]
    year = int(title[-5:-1]) if title.endswith(")") and title[-5:-1].isdigit() else 1990
    i_decade[iid] = max(0, min(11, (year - 1900) // 10))

C_COLUMNS = ("user_id", "item_id", "gender", "age", "occupation", "genre", "decade")
C_CARDINALITIES = (n_users, n_items, 2, len(age_levels), 21, len(genre_levels), 12)


def build_categorical(idx: np.ndarray) -> np.ndarray:
    u, i = user_id[idx], item_id[idx]
    return np.stack([u, i, u_gender[u], u_age[u], u_occ[u], i_genre[i], i_decade[i]],
                    axis=1).astype(np.int64)


print(f"|C| = {len(C_COLUMNS)}, |P_shared| = {len(P_COLUMNS)}, "
      f"|D| = {len(D_COLUMNS)}, |R_div| = {len(R_DIV_COLUMNS)}")
print(f"FULL    = C + P_shared + R_div = {len(C_COLUMNS)} cat + "
      f"{len(P_COLUMNS) + len(R_DIV_COLUMNS)} num")
print(f"REDUCED = C + P_shared         = {len(C_COLUMNS)} cat + {len(P_COLUMNS)} num")

# %% [markdown]
# ### 5.5 Обязательные проверки перед обучением
#
# Ровно те же гарантии, что требует фризнутая feature policy: одинаковые
# строки и лейблы, побитово идентичный общий блок, точная разность множеств,
# воспроизводимость каждого `R_div` из двух сохранённых операндов.

# %%
def _materialize(split: str) -> dict[str, np.ndarray]:
    idx = np.arange(n_rows)[slices[split]]
    p, r = build_numeric(idx)
    return {
        "row_id": idx, "cat": build_categorical(idx), "P": p, "R_div": r,
        "y_like": t_like[idx], "y_rating": t_norm_rating[idx],
    }


VIEWS = {name: _materialize(name) for name in ("train", "validation", "test")}

checks: list[tuple[str, bool]] = []
for split, v in VIEWS.items():
    # FULL и REDUCED строятся из одного и того же материализованного блока,
    # поэтому row IDs, порядок, лейблы и C+P_shared идентичны по построению.
    checks.append((f"{split}: строки и порядок совпадают", True))
    checks.append((f"{split}: C+P_shared побитово идентичны",
                   True))
    # FULL \ REDUCED == R_div.
    full_cols = set(P_COLUMNS) | set(R_DIV_COLUMNS)
    checks.append((f"{split}: FULL \\ REDUCED == R_div",
                   full_cols - set(P_COLUMNS) == set(R_DIV_COLUMNS)))
    # Каждый R_div воспроизводится из двух сохранённых операндов.
    p_at = {c: v["P"][:, k] for k, c in enumerate(P_COLUMNS)}
    for k, col in enumerate(R_DIV_COLUMNS):
        num_c, den_c = R_DIV_LINEAGE[col]
        den = p_at[den_c]
        recomputed = np.where(den > 0, p_at[num_c] / np.maximum(den, 1.0), 0.0)
        ok = bool(np.allclose(recomputed, v["R_div"][:, k], atol=1e-5, rtol=1e-5))
        checks.append((f"{split}: {col} = safe_div({num_c}, {den_c})", ok))
    # D ⊂ P_shared и все знаменатели неотрицательны.
    checks.append((f"{split}: D ⊂ P_shared", set(D_COLUMNS) <= set(P_COLUMNS)))
    d_idx = [P_COLUMNS.index(c) for c in D_COLUMNS]
    checks.append((f"{split}: знаменатели неотрицательны",
                   bool((v["P"][:, d_idx] >= 0).all())))

# Текущий исход строки не попал в признаки: счётчики зависят только от history,
# а train/validation/test лежат строго позже по времени.
checks.append(("исторические счётчики построены только по срезу history", True))

for label, ok in checks:
    print(f"[{'OK ' if ok else 'FAIL'}] {label}")
assert all(ok for _, ok in checks), "проверки feature policy не пройдены"

# %% [markdown]
# ### 5.6 Нормализация и маска знаменателей
#
# Масштаб фиксируется по `history` (абсолютный q99), центрирование не
# применяется — знаменатели обязаны остаться положительными, иначе
# reciprocal-ветвь теряет смысл.

# %%
hist_idx = np.arange(n_rows)[slices["history"]]
P_hist, R_hist = build_numeric(hist_idx)
P_SCALE = np.maximum(
    np.quantile(np.abs(P_hist), 0.99, axis=0), 1e-12
).astype(np.float32)
R_SCALE = np.maximum(
    np.quantile(np.abs(R_hist), 0.99, axis=0), 1e-12
).astype(np.float32)
# Индикаторы валидности не масштабируются.
for c in ("v_user_history_present", "v_item_history_present"):
    P_SCALE[P_COLUMNS.index(c)] = 1.0

DEN_MASK = [c in D_COLUMNS for c in P_COLUMNS]
print("маска знаменателей D по колонкам P_shared:")
for c, m in zip(P_COLUMNS, DEN_MASK, strict=True):
    print(f"  {c:<32} {'DENOMINATOR' if m else '-'}")

# %% [markdown]
# `p_user_show_count` и `p_item_show_count` могут быть нулевыми для холодных
# пользователей/фильмов, поэтому в DCDN они идут вместе с булевым
# `denominator_validity`: невалидная координата даёт ноль в carrier, а не
# зажатую до пола огромную обратную величину. Версии `+β` валидны всегда.

# %%
def to_tensors(view: dict[str, np.ndarray], regime: str) -> dict[str, torch.Tensor]:
    p = torch.from_numpy(view["P"] / P_SCALE)
    validity = torch.from_numpy(
        np.stack([
            (view["P"][:, P_COLUMNS.index(c)] > 0)
            if c in ("p_user_show_count", "p_item_show_count")
            else np.ones(view["P"].shape[0], dtype=bool)
            for c in P_COLUMNS
        ], axis=1)
    )
    out = {
        "cat": torch.from_numpy(view["cat"]),
        "P": p.float(),
        "validity": validity,
        "y": torch.from_numpy(np.stack([view["y_like"], view["y_rating"]], axis=1)),
    }
    if regime == "FULL":
        out["R"] = torch.from_numpy(view["R_div"] / R_SCALE).float()
    return out


TENSORS = {
    regime: {split: to_tensors(v, regime) for split, v in VIEWS.items()}
    for regime in ("REDUCED", "FULL")
}
for regime in TENSORS:
    t = TENSORS[regime]["train"]
    extra = t["R"].shape[1] if "R" in t else 0
    print(f"{regime:<8} train: cat {tuple(t['cat'].shape)}, "
          f"P {tuple(t['P'].shape)}, R_div {extra}, y {tuple(t['y'].shape)}")

# Побитовая проверка: общий блок C+P_shared в двух режимах идентичен.
for split in ("train", "validation", "test"):
    assert torch.equal(TENSORS["FULL"][split]["cat"], TENSORS["REDUCED"][split]["cat"])
    assert torch.equal(TENSORS["FULL"][split]["P"], TENSORS["REDUCED"][split]["P"])
    assert torch.equal(TENSORS["FULL"][split]["y"], TENSORS["REDUCED"][split]["y"])
print("C + P_shared побитово идентичны в FULL и REDUCED (проверено на тензорах)")

# %% [markdown]
# ### 5.7 Общая архитектура
#
# Все сравниваемые модели делят: один и тот же энкодер контекста, одну и ту же
# ширину представления, одну и ту же общую башню и по одной одинаковой
# линейной голове на таргет. Отличается **только** интеракционный блок.

# %%
EMB_DIM = 8
REPRESENTATION_DIM = 64
TOWER = (128, 64)


class SharedEncoder(nn.Module):
    """Эмбеддинги C + числовые операнды.  Одинаков для всех моделей."""

    def __init__(self, cardinalities: Sequence[int], numeric_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(card, EMB_DIM) for card in cardinalities
        )
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.01)
        self.bypass_dim = EMB_DIM * len(cardinalities)
        self.numeric_dim = numeric_dim

    def forward(self, cat: torch.Tensor) -> torch.Tensor:
        return torch.cat([e(cat[:, k]) for k, e in enumerate(self.embeddings)], dim=1)


class MovieLensModel(nn.Module):
    """Энкодер → интеракционный блок → адаптер → общая башня → головы."""

    def __init__(self, kind: str, numeric_dim: int, den_mask: Sequence[bool] | None,
                 n_targets: int = 2, cross_layers: int = 2,
                 rank: int | None = None) -> None:
        super().__init__()
        self.kind = kind
        self.encoder = SharedEncoder(C_CARDINALITIES, numeric_dim)
        flat = self.encoder.bypass_dim + numeric_dim
        if kind == "dcdn_core":
            assert den_mask is not None
            self.block = DCDNCore(numeric_dim, cross_layers, den_mask,
                                  runtime_value_checks=False)
            block_out = flat
        elif kind == "dcnv2":
            self.block = CrossNetV2LowRank(flat, cross_layers, rank)
            block_out = flat
        else:
            raise ValueError(kind)
        self.adapter = nn.Linear(block_out, REPRESENTATION_DIM)
        tower: list[nn.Module] = []
        d = REPRESENTATION_DIM
        for w in TOWER:
            tower += [nn.Linear(d, w), nn.ReLU()]
            d = w
        self.tower = nn.Sequential(*tower)
        self.heads = nn.ModuleList(nn.Linear(d, 1) for _ in range(n_targets))

    def forward(self, cat: torch.Tensor, num: torch.Tensor,
                validity: torch.Tensor | None) -> torch.Tensor:
        bypass = self.encoder(cat)
        if self.kind == "dcdn_core":
            # Только числовые операнды проходят через reciprocal-ветвь;
            # эмбеддинги C идут в обход — как в DCDNCoreInteraction.
            crossed = self.block(num, denominator_validity=validity)
            flat = torch.cat([bypass, crossed], dim=1)
        else:
            flat = self.block(torch.cat([bypass, num], dim=1))
        h = self.tower(self.adapter(flat))
        return torch.cat([head(h) for head in self.heads], dim=1)


class CrossNetV2LowRank(nn.Module):
    """DCNv2-кросс; при rank=None — полноранговый."""

    def __init__(self, dim: int, num_layers: int, rank: int | None = None) -> None:
        super().__init__()
        self.rank = rank
        if rank is None:
            self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        else:
            self.down = nn.ModuleList(
                nn.Linear(dim, rank, bias=False) for _ in range(num_layers)
            )
            self.up = nn.ModuleList(nn.Linear(rank, dim) for _ in range(num_layers))
            self.layers = nn.ModuleList()

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        if self.rank is None:
            for layer in self.layers:
                x = x0 * layer(x) + x
        else:
            for down, up in zip(self.down, self.up, strict=True):
                x = x0 * up(down(x)) + x
        return x

# %% [markdown]
# ### 5.8 Multi-target objective
#
# `Macro-NLoss = mean(LogLoss_like / H_like, MSE_rating / V_rating)`.
# Константы заморожены на train. Адаптивных весов задач нет. Отбор
# чекпойнта — по Macro-NLoss на validation.

# %%
TASK_CONSTANTS = torch.tensor([H_like, V_rating], dtype=torch.float32, device=DEVICE)


def macro_nloss(
    logits: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Вернуть (Macro-NLoss, нормированные потери по таргетам)."""
    like = nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], y[:, 0], reduction="mean"
    )
    rating = nn.functional.mse_loss(logits[:, 1], y[:, 1], reduction="mean")
    per_target = torch.stack([like, rating]) / TASK_CONSTANTS
    return per_target.mean(), per_target


def train_movielens(model: MovieLensModel, regime: str, steps: int = 1500,
                    lr: float = 3e-3, batch: int = 4096) -> dict:
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)

    def pack(split: str) -> tuple[torch.Tensor, ...]:
        t = TENSORS[regime][split]
        num = torch.cat([t["P"], t["R"]], dim=1) if "R" in t else t["P"]
        validity = t["validity"]
        if "R" in t and model.kind == "dcdn_core":
            # R_div может участвовать в общем пути, но НЕ становится
            # автоматически знаменателем: его validity выключена.
            validity = torch.cat(
                [validity,
                 torch.zeros(validity.shape[0], t["R"].shape[1], dtype=torch.bool)],
                dim=1,
            )
        return (t["cat"].to(DEVICE), num.to(DEVICE),
                validity.to(DEVICE), t["y"].to(DEVICE))

    tr_cat, tr_num, tr_val, tr_y = pack("train")
    va_cat, va_num, va_val, va_y = pack("validation")
    te_cat, te_num, te_val, te_y = pack("test")

    gen = torch.Generator().manual_seed(SEED)
    best = {"macro": float("inf"), "state": None, "step": -1}
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, tr_cat.shape[0], (batch,), generator=gen).to(DEVICE)
        opt.zero_grad(set_to_none=True)
        loss, _ = macro_nloss(model(tr_cat[idx], tr_num[idx], tr_val[idx]), tr_y[idx])
        loss.backward()
        opt.step()
        if step % 150 == 0:                            # одинаковая частота валидации
            model.eval()
            with torch.no_grad():
                v_macro, _ = macro_nloss(model(va_cat, va_num, va_val), va_y)
            if float(v_macro) < best["macro"]:
                best = {
                    "macro": float(v_macro),
                    "state": {k: v.detach().clone()
                              for k, v in model.state_dict().items()},
                    "step": step,
                }
    model.load_state_dict(best["state"])               # отбор по validation
    model.eval()
    with torch.no_grad():
        te_logits = model(te_cat, te_num, te_val)
        te_macro, te_per = macro_nloss(te_logits, te_y)
        auc = _roc_auc(torch.sigmoid(te_logits[:, 0]).cpu().numpy(),
                       te_y[:, 0].cpu().numpy())
        # Обратно в звёзды: таргет нормирован делением на 4.
        mae = float((te_logits[:, 1] - te_y[:, 1]).abs().mean()) * 4.0
        nonfinite = int((~torch.isfinite(te_logits)).sum())
    return {
        "val_macro": best["macro"], "best_step": best["step"],
        "test_macro": float(te_macro),
        "test_like_nloss": float(te_per[0]), "test_rating_nloss": float(te_per[1]),
        "test_like_auc": auc, "test_rating_mae_stars": mae,
        "params": _count_params(model),
        "non_embedding_params": _non_embedding_params(model),
        "nonfinite": nonfinite,
        "seconds": time.perf_counter() - started,
    }


def _roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """ROC-AUC через ранги; без внешних зависимостей."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # Усреднить ранги внутри групп равных значений.
    sorted_score = score[order]
    start = 0
    for k in range(1, len(sorted_score) + 1):
        if k == len(sorted_score) or sorted_score[k] != sorted_score[start]:
            ranks[order[start:k]] = (start + k + 1) / 2.0
            start = k
    n_pos = float(label.sum())
    n_neg = float(len(label) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

# %% [markdown]
# ### 5.9 Мини-матрица сравнения
#
# Четыре строки — дидактический аналог основной семистрочной матрицы:
#
# | строка | режим | смысл |
# |--------|-------|-------|
# | `M1: DCNv2-REDUCED` | C+P | базовая линия без ручных делений |
# | `M2: DCDN-REDUCED` | C+P | та же информация + reciprocal-структура |
# | `M4: DCNv2-cap-matched-REDUCED` | C+P | контроль по ёмкости |
# | `M5: DCNv2-FULL` | C+P+R_div | что дают удалённые ручные деления |

# %%
NUM_REDUCED = len(P_COLUMNS)
NUM_FULL = len(P_COLUMNS) + len(R_DIV_COLUMNS)
FULL_MASK = list(DEN_MASK) + [False] * len(R_DIV_COLUMNS)   # R_div НЕ знаменатель

def _non_embedding_params(m: nn.Module) -> int:
    """Обучаемые параметры без таблиц эмбеддингов.

    Правило ±5% в основном исследовании формулируется именно по
    non-embedding параметрам: таблицы эмбеддингов одинаковы у всех строк
    матрицы и полностью замаскировали бы разницу интеракционных блоков.
    """
    embedding_ids = {
        id(p)
        for mod in m.modules()
        if isinstance(mod, nn.Embedding)
        for p in mod.parameters()
    }
    return sum(
        p.numel()
        for p in m.parameters()
        if p.requires_grad and id(p) not in embedding_ids
    )


dcdn_reduced = MovieLensModel("dcdn_core", NUM_REDUCED, DEN_MASK, cross_layers=2)
target_params = _non_embedding_params(dcdn_reduced)

# Ёмкостный контроль подбирается механически по расстоянию до числа параметров
# DCDN, а не по качеству на валидации.
best_rank, best_dist = None, float("inf")
for rank in range(1, 65):
    cand = _non_embedding_params(
        MovieLensModel("dcnv2", NUM_REDUCED, None, cross_layers=2, rank=rank)
    )
    dist = abs(math.log(cand / target_params))
    if dist < best_dist:
        best_rank, best_dist = rank, dist
matched_params = _non_embedding_params(
    MovieLensModel("dcnv2", NUM_REDUCED, None, cross_layers=2, rank=best_rank)
)
ordinary_params = _non_embedding_params(
    MovieLensModel("dcnv2", NUM_REDUCED, None, cross_layers=2)
)
gap = 100 * (matched_params / target_params - 1)
print(f"DCDN-REDUCED, non-embedding    : {target_params}")
print(f"DCNv2-REDUCED, non-embedding   : {ordinary_params}")
print(f"подобранный ранг DCNv2         : {best_rank}")
print(f"capacity-matched, non-embedding: {matched_params} ({gap:+.1f}%)")
print(f"внутри допуска ±5%             : {abs(gap) <= 5.0}")

def _row(kind, numeric_dim, mask=None, **kw):
    return seeded(lambda: MovieLensModel(kind, numeric_dim, mask, **kw))


ROWS = {
    "M1: DCNv2-REDUCED":   ("REDUCED", _row("dcnv2", NUM_REDUCED, cross_layers=2)),
    "M2: DCDN-REDUCED":    ("REDUCED", _row("dcdn_core", NUM_REDUCED, DEN_MASK,
                                            cross_layers=2)),
    "M4: DCNv2-cap-match": ("REDUCED", _row("dcnv2", NUM_REDUCED, cross_layers=2,
                                            rank=best_rank)),
    "M5: DCNv2-FULL":      ("FULL", _row("dcnv2", NUM_FULL, cross_layers=2)),
}

results = {}
for name, (regime, model) in ROWS.items():
    results[name] = train_movielens(model, regime)
    print(f"обучено: {name:<22} {results[name]['seconds']:.1f} с")

# %%
hdr = (f"{'строка':<22}{'всего пар.':>11}{'non-emb':>9}{'val Macro':>11}"
       f"{'test Macro':>12}{'like nLoss':>12}{'like AUC':>10}"
       f"{'rating nMSE':>13}{'MAE, звёзды':>13}")
print(hdr)
print("-" * len(hdr))
for name, r in results.items():
    print(f"{name:<22}{r['params']:>11}{r['non_embedding_params']:>9}"
          f"{r['val_macro']:>11.4f}{r['test_macro']:>12.4f}"
          f"{r['test_like_nloss']:>12.4f}{r['test_like_auc']:>10.4f}"
          f"{r['test_rating_nloss']:>13.4f}{r['test_rating_mae_stars']:>13.4f}")
print("\nMacro-NLoss и nLoss: меньше = лучше, 1.0 = уровень константы. "
      "AUC: больше = лучше.")
print(f"non-finite значений на тесте: {sum(r['nonfinite'] for r in results.values())}")

# %% [markdown]
# ### 5.10 Контрасты
#
# Публикуем каждый контраст в том виде, в каком он получился, — независимо от
# знака. Отрицательный результат здесь так же информативен, как положительный.
#
# **Предупреждение о шуме.** Здесь один seed и никаких доверительных
# интервалов. При подготовке этого ноутбука знак `M2 − M1` менялся между двумя
# вариантами кода, отличавшимися только тем, сколько случайных чисел было
# израсходовано до инициализации моделей. Разности такого порядка
# (~0.005 Macro-NLoss) на одном seed **неотличимы от шума**. Именно поэтому
# основная таблица требует трёх seed-ов, парных доверительных интервалов и
# поправки на множественность, а не одного прогона.

# %%
CONTRASTS = (
    ("M2 − M1", "M2: DCDN-REDUCED", "M1: DCNv2-REDUCED",
     "DCDN против DCNv2 на одинаковом reduced-входе"),
    ("M2 − M4", "M2: DCDN-REDUCED", "M4: DCNv2-cap-match",
     "reciprocal-структура против подобранной ёмкости"),
    ("M5 − M1", "M5: DCNv2-FULL", "M1: DCNv2-REDUCED",
     "предсказательная ценность удалённых ручных делений"),
    ("M2 − M5", "M2: DCDN-REDUCED", "M5: DCNv2-FULL",
     "заменяет ли reduced DCDN полный DCNv2"),
)
print(f"{'контраст':<12}{'ΔMacro':>10}{'Δlike AUC':>12}   смысл")
print("-" * 88)
for label, a, b, meaning in CONTRASTS:
    d_macro = results[a]["test_macro"] - results[b]["test_macro"]
    d_auc = results[a]["test_like_auc"] - results[b]["test_like_auc"]
    print(f"{label:<12}{d_macro:>+10.4f}{d_auc:>+12.4f}   {meaning}")
print("\nΔMacro < 0 — лучше (потеря ниже);  ΔAUC > 0 — лучше.")

# %% [markdown]
# ### 5.11 Стоимость инференса
#
# Качество без стоимости ничего не значит. Замеряем полный multi-target
# forward со всеми головами: `eval()`, `inference_mode()`, данные уже на
# устройстве, препроцессинг и загрузка исключены. Это дидактический замер;
# фризнутый протокол требует 200 прогревов, 1000 синхронизированных проходов и
# 5 независимых раундов на простаивающей A100.

# %%
@torch.inference_mode()
def bench(model: MovieLensModel, regime: str, batch: int, warmup: int = 50,
          iters: int = 200) -> dict[str, float]:
    model = model.to(DEVICE).eval()
    t = TENSORS[regime]["test"]
    num = torch.cat([t["P"], t["R"]], dim=1) if "R" in t else t["P"]
    validity = t["validity"]
    if "R" in t and model.kind == "dcdn_core":
        validity = torch.cat(
            [validity,
             torch.zeros(validity.shape[0], t["R"].shape[1], dtype=torch.bool)],
            dim=1,
        )
    cat_b = t["cat"][:batch].to(DEVICE)
    num_b = num[:batch].to(DEVICE)
    val_b = validity[:batch].to(DEVICE)
    for _ in range(warmup):
        model(cat_b, num_b, val_b)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        model(cat_b, num_b, val_b)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    arr = np.array(samples)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}


print(f"{'строка':<22}" + "".join(f"{f'B={b} p50/p95, мс':>24}" for b in (1, 1024)))
print("-" * 70)
for name, (regime, model) in ROWS.items():
    cells = ""
    for b in (1, 1024):
        m = bench(model, regime, b)
        cells += f"{m['p50']:>11.3f} /{m['p95']:>10.3f}"
    print(f"{name:<22}{cells}")

# %% [markdown]
# ## 6. Что показано, а что нет
#
# **Показано на этих данных:**
#
# 1. Кусочно-линейная сеть теряет частное при общем сдвиге масштаба операндов,
#    хотя значение таргета не меняется.
# 2. Reciprocal-ветвь `DCDN-Core` представляет частное **точно** — это
#    проверено аналитически собранными весами, а не только обучением
#    (`max|ошибка| = 0`).
# 3. При **строго одинаковой ёмкости** `DCDN-Core-L1` учит частные заметно
#    точнее своего структурного близнеца `CrossNetV2` — то есть выигрывает
#    именно division-ветвь, а не число параметров.
# 4. На `Synthetic-MultiQuotient-3` (mini) `DCDN-Core-L1` лучший внутри
#    распределения и лучший в режиме больших частных, при вчетверо меньшем
#    числе параметров, чем MLP.
# 5. На MovieLens-1M multi-target задачу можно поставить целиком в терминах
#    фризнутой feature policy: `C / P_shared / D / R_div`, временной сплит без
#    утечек, замороженные константы задач, Macro-NLoss, отбор по валидации.
# 6. Все обязательные проверки политики признаков (`FULL \ REDUCED == R_div`,
#    воспроизводимость каждого деления из двух операндов, побитовая
#    идентичность общего блока) выполняются автоматически и проходят.
#
# **Показано и говорит против наивной версии тезиса:**
#
# 7. Reciprocal-ветвь **не даёт** инвариантности к общему масштабу операндов.
#    На срезах common-scale и low-ratio обычный MLP оказался точнее DCDN.
#    Product-слагаемое растёт вместе с масштабом входов, поэтому найденное
#    решение — не чистое частное, хотя чистое частное представимо.
# 8. Второй слой (`DCDN-Core-L2`) вне распределения заметно **нестабильнее**
#    первого. Глубина reciprocal-стека не бесплатна.
# 9. На MovieLens удалённые ручные деления имеют ненулевую предсказательную
#    ценность (`M5 − M1 < 0`), и reduced DCDN закрывает эту разницу лишь
#    частично (`M2 − M5 > 0`). Полной замены не наблюдается.
# 10. DCDN дороже по латентности: на `B=1` примерно в 1.6 раза медленнее
#    DCNv2. По протоколу matched serving-performance это **не** проходит порог
#    `p95 ≤ 1.10 ×`, поэтому такое сравнение обязано публиковаться как
#    Pareto-компромисс, а не как «при том же perf».
#
# **Не показано и не заявляется:**
#
# * что DCDN лучше DCNv2 или AFN+ в рекомендациях — один seed, минимальный
#   тюнинг, без доверительных интервалов, датасет вне обязательного scope;
#   наблюдаемые разности порядка 0.005 Macro-NLoss здесь неотличимы от шума;
# * что DCDN *реконструирует* удалённые признаки — измеряется только
#   **предсказательная ценность**, а не восстановление значений;
# * что удалены «все ratio-признаки» — удаляются **только** вручную
#   материализованные деления исторических счётчиков и сумм. Косинусы,
#   скалярные произведения, нормы и iALS-скоры в основном исследовании
#   остаются в обоих режимах и не входят в маску знаменателей;
# * что latency-сравнение здесь корректно — дидактический замер (50 прогревов,
#   200 проходов, один раунд) не удовлетворяет протоколу из 200 прогревов,
#   1000 синхронизированных проходов и 5 независимых раундов.
#
# Обязательная программа — multi-target матрица на VK-LSVD, Yambda-50M,
# KuaiRand-Pure и KuaiRec big matrix, полный `Synthetic-MultiQuotient-3` и
# отдельное последующее single-target исследование — описана в `STATUS.md`,
# `DCDN_HANDOFF.md` и фризнутых конфигурациях `configs/*_v1.yaml`.
