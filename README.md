# capstone-ai
This is a repo of my exploratory scripts for my Capstone project: AI Learns to play atari 2600 'Assault':

## Installation
The scripts require several packages, some requiring different versions.
For convenience, I separated each of my research ideas in separate folders.

BASELINES requires:
tensorflow==1.15.0
stable-baselines>=2.10.2


Which can be installed by:
```bash
pip install tensorflow==1.15.0
pip install stable-baselines==2.10.2
```



DQN requires:
tensorflow==2.4.0
tqdm>=4.60.0
gym
tqdm


Which can be installed by:
```bash
pip install tensorflow==2.4.0
pip install tqdm==4.60.0
pip install gym
pip install tqdm
```



ENV_DEMO requires:
tensorflow==2.4.0
gym

Which can be installed by:
```bash
pip install tensorflow==2.4.0
pip install gym
```



NEAT requires:
numpy
neat-python
gym

Which can be installed by:
```bash
pip install numpy
pip install neat-python
pip install gym
```



VISUALIZE requires:
pandas
scipy
matplotlib

Which can be installed by:
```bash
pip install pandas
pip install scipy
pip install matplotlib
```

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

