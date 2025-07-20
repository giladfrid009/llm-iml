## Installation

To install the dependencies of this project, pleaese use the ***uv*** dependency manager. (highly recommended)   
*Note:* If you dont have ***uv*** installed, see docs at the [official website](https://docs.astral.sh/uv/getting-started/installation/).

#### Dependencies:

The dependencies are located in two files:
* ```pyproject.toml``` - contains the general package names and versions, should be enough for standard installations.
* ```uv.lock``` - contains exact versions of all installed packages. 

#### Steps:

1. Clone Repo from github
2. Navigate to the repo folder
3. Run the following command in your terminal: ```uv sync```

#### Env Activation:

1. Navigate to the repo folder
2. Run the following command in your terminal: ```source ./.venv/bin/activate```
