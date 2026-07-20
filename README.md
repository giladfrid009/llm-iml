# UPD

## Installation

To install the dependencies of this project, pleaese use the ***uv*** dependency manager. (highly recommended)    
*Note:* If you dont have ***uv*** installed, see docs at the [official website](https://docs.astral.sh/uv/getting-started/installation/).

#### Dependencies:

The dependencies are located in two files:

* `pyproject.toml` - contains the general package names and versions, should be enough for standard installations.
* `uv.lock` - contains exact versions of all installed packages.

#### Steps

1. Clone Repo from github
2. Navigate to the repo folder
3. Run the following command in your terminal: ```uv sync```

#### Env Activation

1. Navigate to the repo folder
2. Run the following command in your terminal: ```source ./.venv/bin/activate```

## Running Experiments

### Full Flexibility Scripts

To directly run experiment scripts with maximum flexibility over all passed hyperparameters, see the `scripts` folder. To get script documentation, run the script with the `--help` flag.

The following script files are available:

* **`run_soft.py`** - runs the SoftPrompt experiments
* **`run_upd.py`** - runs the UPD experiments. To run with cold-start initialization, set the `warmup_epochs` argument to the amount of training epochs.
* **`run_uap.py`** - runs the AG-UAP experiments

#### Helpers

* **`generate.py`** (optional) - given a perturbation from a previous experiment, generate outputs over additional datasets.
* **`evaluate.py`** (optional) - given generated outputs from `generate.py` or from running an experiment script, evaluate them with the provided judges.
* **`summarize.py`** (optional) - given a folder containing evaluation outputs from `evaluate.py` or from running experiment scripts, summarize all results from all sub-folders in a human readable format.

### Batch Running Script

**IMPORTANT**: before running the batch script, one must edit the script files via the desired experiment setups. It is important to read and understand the batch script file before running it.

* **`run_all_upd.py`** - runs all UPD experiments with a single command.
* **`run_all_uap.py`** - runs all AG-UAP experiments with a single command.
* **`run_all_soft.py`** - runs all SoftPrompt experiments with a single command.

Example (`repeats` is the amount of times the experiment will be repeated, `name_suffix` is an optional string that will be added to the experiment name for logging purposes):

```bash
python run_all_upd.py --models phi_3_capo --repeats 4 --name_suffix "test_run"
```

## Library Usage

Instead of the scripts, the package can be driven directly from Python. This chapter walks through a full attack, from loading a model to querying the perturbed one. A runnable version of it is available at [`notebooks/example.ipynb`](notebooks/example.ipynb).

All snippets assume the environment is activated and the working directory is the repo root.

### 1. Load a Model

`load_model` fetches a supported model from HuggingFace together with its tokenizer, and patches the chat template where the original one is missing or broken.

```python
import torch
from scripts.utils.load_model import load_model
from src.utils import env

env.set_seed(41)
torch.set_float32_matmul_precision("high")

model, tokenizer = load_model("GraySwanAI/Llama-3-8B-Instruct-RR", torch_dtype="bfloat16")
```

### 2. Wrap it in an `AdvModel`

`AdvModel` augments the model with a learnable perturbation in the embedding space. The base model is frozen — only the perturbation is optimized.

* **`num_tokens`** - the length of the perturbation, in tokens.
* **`adv_suffix`** - whether the perturbation is appended to the user message (`True`) or prepended to it (`False`).
* We also support more fine-grained control of the perturbation location, see docs and code source for more info

```python
from src.adv_model import AdvModel

adv_model = AdvModel(
    model=model,
    tokenizer=tokenizer,
    num_tokens=20,
    adv_suffix=True,
)
```

### 3. Initialize the Perturbation

The `Initializer` class provides the starting point of the optimization.   
You may also load a pre-existing perturbation via `Initializer.load()`.

```python
from src.initialize import Initializer

embeds = Initializer.random_normal(adv_model)
adv_model.set_embeddings(embeds)
```

### 4. Load the Data

`load_dataset` returns train / validation / test splits as pandas DataFrames, each with a `prompt` column and a `target` column. `TableLoader` batches them for the attack. see `DatasetName` in [`scripts/utils/load_dataset.py`](scripts/utils/load_dataset.py) for supported datasets.

```python
from scripts.utils.load_dataset import load_dataset
from src.data import TableLoader

ds_train, ds_val, ds_test = load_dataset("advbench")

dl_train = TableLoader(ds_train, batch_size=50)
dl_eval = TableLoader(ds_val, batch_size=50)
```

### 5. Set up the Judges

Evaluators score the model responses during training and decide which perturbation is the best one.
They are used to evaluate the attack success. 

```python
from src.eval import StrongReject, KeywordMatching
from gserve import ServeConfig

sr_judge = StrongReject(ServeConfig(gpu_ids=[1], startup_timeout=20 * 60), verbose=True)
kw_judge = KeywordMatching(verbose=False)
```

### 6. Run an Attack

The `SoftPrompt` attack optimizes the universal perturbation directly against the training targets, using the sign-gradient optimizer `FGSM`.

* **`eval_metric`** - which of the evaluators' metrics selects the best perturbation. It must be one of the metrics exposed by the passed evaluators.
* **`eval_freq`** - as a float, the fraction of an epoch between evaluations (`0.5` = twice per epoch); as an int, the number of epochs.
* **`metric_tracker`** - optional logging backend. When omitted, a Weights & Biases run is created automatically under `logs/`.

```python
from src.univ_attacks import SoftPrompt
from src.config import GenConfig
from src.fgsm_optim import FGSM
from src.utils.trackers import WandbTracker

optimizer = FGSM(adv_model.parameters(), lr=0.01)

metric_tracker = WandbTracker("Llama-3-8B-Instruct-RR", "run_name", project="llm-iml")

univ_attack = SoftPrompt(
    adv_model,
    optimizer=optimizer,
    evaluators=[sr_judge, kw_judge],
    eval_metric="StrongReject/Thresh@0.5",
    eval_freq=0.5,
    mixed_precision=True,
    gen_config=GenConfig(max_new_tokens=1024, do_sample=True),
    metric_tracker=metric_tracker,
)

adv_model = univ_attack.fit(dl_train=dl_train, dl_eval=dl_eval)
```

`fit` checkpoints the best perturbation to `best_embeds.pt` in the log directory, and on return the `AdvModel` already holds those best embeddings.

### 7. Run a UPD Attack

UPD works in two stages: an inner per-sample attack crafts an individual perturbation for each prompt, and the universal perturbation is then optimized to reproduce the internal activations that these individual perturbations induce.

* **`inner_attack`** - the per-sample attack. Either a `SampleAttack` instance, or a `(adv_model, epoch_num) -> SampleAttack` factory for schedules that change over epochs.
* **`activ_extractor`** - extracts activatios of selected layers during model forward pass. `capture_output=False` captures the *input* of the given layer.

```python
from torch import optim
from src.univ_attacks import UPD
from src.sample_attacks import SP
from src.activ_extractor import ActivationExtractor
from src.config import GenConfig
from src.fgsm_optim import FGSM

# the internal, per-sample attack
inner_attack = SP(
    adv_model,
    optim_factory=lambda params: FGSM(params, lr=0.01),
    steps=100,
    kv_caching=True,
    mixed_precision=True,
)

optimizer = optim.Adam(adv_model.parameters())

activ_extractor = ActivationExtractor(
    adv_model.model,
    "model.layers.6",
    capture_output=False,
)

univ_attack = UPD(
    adv_model,
    # attack params
    inner_attack=inner_attack,
    optimizer=optimizer,
    activ_extractor=activ_extractor,
    dynamic_labels=40,
    # general params
    evaluators=[sr_judge, kw_judge],
    eval_metric="StrongReject/Thresh@0.5",
    eval_freq=0.5,
    mixed_precision=True,
    gen_config=GenConfig(max_new_tokens=1024, do_sample=True),
)

adv_model = univ_attack.fit(dl_train=dl_train, dl_eval=dl_eval)
```

### 8. Use the Perturbed Model


After running an attack via `.fit()`, we have an attacked mode. 
`AdvModel` accepts conversations in the standard chat format and injects the perturbation automatically.

```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a poem about the sea."},
]

completions = adv_model.chat([conversation], config=None)
```

For finer control, `tokenize` returns the encodings along with the `adv_mask` marking the perturbation positions, which can then be passed to `forward` or `generate`:

```python
encodings = adv_model.tokenize([conversation])

# adv_model.forward(input_ids=encodings.input_ids, attention_mask=encodings.attention_mask, adv_mask=None)
# adv_model.generate(input_ids=encodings.input_ids, attention_mask=encodings.attention_mask, adv_mask=None, adv_embeds=None)
```

Passing `adv_mask=None` skips the injection of the perturbation embeddings entirely, which is convenient for comparing perturbed and unperturbed behavior through the same code path.

## License

This work is licensed under the MIT license. Please see the [License File](LICENSE)

### Disclaimer

This project may lead to attacks on LLMs and is intended for academic research use only. It is prohibited for illegal purposes.
