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

```
pip3 install -r requirements.txt
```

4. Download the model to translate text with doc2vec from the
[official website](http://rusvectores.org/ru/models/):
4.1. Optional. If you are behind the proxy, run:
```
export http_proxy=http://user:password@corp-proxy.com:911
```
4.2. Download the model file:
```
curl -O http://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextcbow_300_5_2018.tgz
```

4.3. Unpack the archive:
```
mkdir models/araneum
tar -xvzf araneum_none_fasttextcbow_300_5_2018.tgz -C models/araneum/
```

4.4. Remove obsolete archive:
```
rm araneum_none_fasttextcbow_300_5_2018.tgz
```



## Running

Example command:
```
python3.5 main.py --data-dir data/ \
                  --output-dir output/ \
                  --model models/araneum_none_fasttextcbow_300_5_2018.model
```

If you run behind corporate proxy, use `--proxy`:
```
python3.5 main.py --model models/araneum_none_fasttextcbow_300_5_2018.model\
                  --proxy http://user:password@corp-proxy.com:911
```
