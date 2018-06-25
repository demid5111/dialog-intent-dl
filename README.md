This is the repository for experiments with RNN(LSTM) models applied to analysis of dialogs

## Setting up

1. Create virtual Python environment:
```
virtualenv -p /usr/bin/python3.6 env3
```
OR
```
conda create -n env3 python=3.6
```

2. Activate it:
```
. env3/bin/activate
```

OR

```
source activate env3
```

3. Install dependencies:
3.1 For macOS install:
```
brew install homebrew/science/igraph
```
```
pip3 install -r requirements.txt
```