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

## License

This work is licensed under the MIT license. Please see the [License File](LICENSE)
