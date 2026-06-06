# 連續 Saliency Unlearning 分類專案

本專案基於原論文 SalUn 的官方 Classification 程式碼：
[OPTML-Group/Unlearn-Saliency/Classification](https://github.com/OPTML-Group/Unlearn-Saliency/tree/master/Classification)。

原版 SalUn 主要處理一次性遺忘。本專案將它延伸成「連續類別遺忘」設定：刪除請求會一個接一個到來，模型不只要忘掉當前類別，也要避免已經遺忘過的舊類別在後續訓練中反彈。

## 和原論文專案的主要差異

| 項目 | 原版 SalUn Classification | 本專案 |
| --- | --- | --- |
| 遺忘設定 | 一次性遺忘，透過 `class_to_replace`、`indexes_to_replace` 或預先產生的 `mask_path` 指定遺忘資料。 | 連續類別遺忘，透過 `--forget_sequence` 指定遺忘順序，例如 `0,1,2,3,4`。 |
| Mask 記憶 | 針對單次遺忘任務使用一個 saliency mask。 | 使用命中次數矩陣（Hit-Count Matrix），記錄每個參數跨輪被 mask 選中的次數。 |
| 參數分區 | 參數只有「被 mask 選中」與「沒被 mask 選中」兩種狀態。 | 每輪將參數切成 `new`、`conflict`、`old`、`other` 四種區域。 |
| Conflict 區域力道 | 沒有針對連續遺忘重複命中的頻率衰減。 | 衝突區域（Conflict Region）使用 `alpha_conflict ** hit_count`，例如 `0.5^n`，降低共用特徵被反覆推擠的傷害。 |
| Optimizer 控制方式 | 原版 SalUn 主要使用 gradient mask 與參數 restore 邏輯。 | 使用更新後修正（post-update correction）：先讓 optimizer 正常更新，再依照區域策略重寫每個參數的實際更新幅度。 |
| 執行環境 | 原始碼中多處假設 CUDA。 | 改成集中選擇 CUDA、Apple MPS 或 CPU，方便在 MBP 或無 CUDA 環境測試。 |
| 套件與本機檔案 | 原版 requirements。 | 補上 `pyarrow==12.0.1` 以相容目前 `datasets` 版本，並忽略 `.python`、`.venv`、`data`、`checkpoints` 等本機檔案。 |

## 程式碼改動地圖

可以用以下指令搜尋本專案特別標示的改動區塊：

```bash
rg "PROJECT MOD"
```

主要改動如下：

- `main_forget.py`
  - 將原本的一次性遺忘流程改成 `--forget_sequence` 控制的連續遺忘迴圈。
  - 每一輪都重新建立 `forget_loader` 和 `retain_loader`。
  - 每一輪都根據當前遺忘類別重新計算 saliency mask。
  - 每一輪結束後更新 historical hit-count mask。

- `utils_mask.py`
  - 新增檔案，原版 SalUn 沒有這個檔案。
  - `update_historical_mask()`：累積每個參數被 mask 選中的次數。
  - `partition_regions()`：切分 `new`、`conflict`、`old`、`other`，並附上 `hit_count`。

- `generate_mask.py`
  - 保留原版 standalone mask generation 流程。
  - 新增 `generate_mask_for_class()`，讓 `main_forget.py` 可以在每一輪直接產生當前類別的 mask。

- `unlearn/RL.py`
  - 讓 RL 可以接收 `regions` 和 `alpha_conflict`。
  - 新增 `post_update_correction()`：
    - 新區域（New Region）：`1.0x`
    - 衝突區域（Conflict Region）：`alpha_conflict ** hit_count`
    - 舊區域（Old Region）：`0.0x`
    - 其他區域（Other Region）：`0.0x`

- `unlearn/impl.py`
  - 將 `regions` 和 `alpha_conflict` 從 wrapper 傳進真正的 unlearning epoch function。

- `arg_parser.py`
  - 新增：
    - `--forget_sequence`
    - `--mask_ratio`
    - `--alpha_conflict`

- `utils.py`、`main_train.py`、`main_random.py`、`trainer/train.py`、`trainer/val.py`、`evaluation/SVC_MIA.py`、多個 `unlearn/*`
  - 將原本偏 CUDA-only 的 tensor movement 改成可依照實際環境選擇 CUDA、Apple MPS 或 CPU。

## 演算法流程

對 `--forget_sequence` 中的每個類別，依序執行：

1. 動態切分資料集。
   - Forget set：只包含當前要遺忘的類別。
   - Retain set：排除當前類別，也排除所有之前已經遺忘過的類別。

2. 產生本輪 saliency mask `M_t`。
   - 根據當前 forget set 的梯度 saliency，選出全模型 top `--mask_ratio` 的參數。

3. 將 `M_t` 與 historical hit-count mask 比對。
   - `new`：現在被選中，以前沒被選中。
   - `conflict`：現在被選中，以前也被選中。
   - `old`：現在沒被選中，以前被選中。
   - `other`：現在和以前都沒被選中。

4. 執行 RL unlearning，並使用更新後修正（post-update correction）。
   - Optimizer 先提出正常更新步伐。
   - 本專案再依照 region policy 重新縮放實際參數變化。
   - 衝突區域（Conflict Region）使用依命中次數衰減（frequency-based decay）：

```text
effective_step = optimizer_step * (alpha_conflict ** hit_count)
```

5. 將本輪 mask 加進 historical hit-count matrix。

## 避免 Adam 與動態學習率衝突的設計

這個設計實作在 `unlearn/RL.py` 的 `post_update_correction()`。

本專案沒有直接在 gradient 階段套用動態學習率，原因是 Adam 這類 optimizer 會使用 momentum 與 adaptive scaling。如果只把 conflict region 的 gradient 乘上很小的倍率，Adam 內部的動量與自適應縮放仍可能改變實際參數更新幅度，導致我們設計的 `0.5^n` 力道控制不夠精準。

因此本專案採用「更新後修正」：

1. 在 `optimizer.step()` 前先備份參數：

```python
theta_before = {name: param.data.clone() ...}
```

2. 讓 optimizer 正常計算並更新：

```python
optimizer.step()
```

3. 更新後計算 optimizer 原本想走的步伐：

```python
delta = after - before
```

4. 依照 region policy 重寫最後參數：

```python
param.data.copy_(before + scale * delta)
```

其中 `scale` 由四種區域決定：

```text
new      -> 1.0
conflict -> alpha_conflict ** hit_count
old      -> 0.0
other    -> 0.0
```

這樣做的效果是：先尊重 optimizer 算出的方向與步伐，再把最後真正落在參數上的位移強制縮放成我們要的比例。也就是說，Adam 可以照常工作，但最終更新幅度仍由本專案的動態分區策略掌控。

## `forget_sequence` 代表什麼

`forget_sequence` 裡面放的是資料集的 class label id，不是圖片 index。

以 CIFAR-100 為例：

```text
0 = apple
1 = aquarium_fish
2 = baby
3 = bear
4 = beaver
```

所以：

```bash
--forget_sequence 0,1,2,3,4
```

代表依序遺忘：

```text
apple -> aquarium_fish -> baby -> bear -> beaver
```

查看完整 CIFAR-100 類別對照：

```bash
.venv/bin/python - <<'PY'
import pickle
with open('./data/cifar-100-python/meta', 'rb') as f:
    meta = pickle.load(f, encoding='latin1')
for i, name in enumerate(meta['fine_label_names']):
    print(i, name)
PY
```

## 環境設定

本專案使用專案內的 Python 3.11 與 `.venv`，不使用 conda。

如果 `.python/` 已經放好專案本地 Python 3.11：

```bash
.python/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

查看目前會使用哪個運算裝置：

```bash
.venv/bin/python -c 'import torch, types, utils; args=types.SimpleNamespace(gpu=0); device=utils.get_device(args); print("selected", device); print("cuda", torch.cuda.is_available()); print("mps", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else None)'
```

## 測試指令

語法檢查：

```bash
.venv/bin/python -m py_compile $(find . -path './.venv' -prune -o -path './.python' -prune -o -name '*.py' -print)
```

Checkpoint forward smoke test：

```bash
.venv/bin/python -c 'import torch, types, utils; from models import model_dict; args=types.SimpleNamespace(gpu=0); device=utils.get_device(args); ckpt=torch.load("checkpoints/resnet18_cifar100/0model_SA_best.pth.tar", map_location=device); model=model_dict["resnet18"](num_classes=100).to(device); model.load_state_dict(ckpt["state_dict"], strict=False); y=model(torch.randn(2,3,32,32,device=device)); print("device", device, "output", tuple(y.shape), y.device)'
```

連續遺忘範例：

```bash
.venv/bin/python main_forget.py \
  --data ./data \
  --dataset cifar100 \
  --num_classes 100 \
  --arch resnet18 \
  --unlearn RL \
  --resume checkpoints/resnet18_cifar100/0model_SA_best.pth.tar \
  --forget_sequence 0,1,2,3,4 \
  --mask_ratio 0.01 \
  --alpha_conflict 0.5
```

## 原版 SalUn 指令

以下原版 one-shot 指令仍可作為 baseline 使用。

訓練原始模型：

```bash
python main_train.py --arch resnet18 --dataset cifar10 --lr 0.1 --epochs 182
```

產生一次性 saliency map：

```bash
python generate_mask.py --save_dir ${saliency_map_path} --model_path ${origin_model_path} --num_indexes_to_replace ${forgetting_data_amount} --unlearn_epochs 1
```

執行一次性 SalUn：

```bash
python main_random.py --unlearn RL --unlearn_epochs ${epochs_for_unlearning} --unlearn_lr ${learning_rate_for_unlearning} --num_indexes_to_replace ${forgetting_data_amount} --model_path ${origin_model_path} --save_dir ${save_dir} --mask_path ${saliency_map_path}
```
