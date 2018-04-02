## TODO
-[x] soft MoE w/ separate decoder
-[x] sampled softmax
-[ ] sampled softmax w/ MoE
-[x] pwv
-[ ] gumble
-[ ] control unit

## Initial Setup

    conda create -n py2 python=2 pip   # create an environment
    source activate py2                # activate environment
    conda install pytorch torchvision -c pytorch # Install requirements

## Prepare data

Divide the data into training, development, and testing splits. Store these files under a directory as follows:

    data/
      |-train.src
      |-train.tgt
      |-dev.src
      |-dev.tgt
      |-test.src
      +-test.tgt
      
Files ending with `.src` is for source language and, `.tgt` is for target language. 

## Quick start

### 1. Preprocess: Prepare vocabulary

Extract vocabulary and counts from the training data
 
    python utils/vocab.py  -dp  /data/train.src  -vp data/vocab.src
    python utils/vocab.py  -dp  /data/train.tgt  -vp data/vocab.tgt

### 2. Training

```bash
python main.py train <data_dir> -c -s src -t tgt  -mp <model_path>
``` 
`-c` flag is used to enable CUDA/GPU.

### 3. Testing

```bash
python main.py test <data_dir> -c -s src -t tgt -mp <model_path>
``` 

some useful flags:
`-bw`: beam width (for testing)
`-ru`: replace UNKs


## Full list of CLI options and default values

```bash
$ python main.py -h
usage: main.py [-h] [-DEBUG] [--seed SEED] [-random] [-cuda] [--beam_width]
               [-replace_unk] [--num_epochs] [--batch_size] [--max_length]
               [--max_size] [--check_frequency] [--validate_frequency] [--msg]
               [--lr_init] [--dropout] [--src] [--tgt] [--model_path]
               [--test_path] [--cell_dim] [--num_layers] [--src_vocab_size]
               [--tgt_vocab_size]
               mode data_dir

positional arguments:
  mode                  mode
  data_dir              Data directory

optional arguments:
  -h, --help            show this help message and exit
  -DEBUG                Debug mode (default: False)
  --seed SEED           Random seed (default: 1234)
  -random, -r           use random random seed (default: False)
  -cuda, -c             Use cuda (default: False)
  --beam_width , -bw    Beam width (default: 1)
  -replace_unk, -ru     Replace unknown tokens (default: False)

Train:
  --num_epochs , -ne    Number of epochs (default: 50)
  --batch_size , -bs    Batch size (default: 64)
  --max_length , -ml    Maximum sequence length (default: 100)
  --max_size , -ms      Maximum training data size (default: 0)
  --check_frequency , -cf
                        How ofter to check progress (default: 100)
  --validate_frequency , -vf
                        How ofter to validate (default: 0)
  --msg , -M            Message (default: None)
  --lr_init             Initial learning rate (default: 0.0005)
  --dropout , -do       dropout rate (default: 0.5)
  --src , -s            Source language (default: de)
  --tgt , -t            Target language (default: en)
  --model_path , -mp    Path to trained model (default: None)

Test:
  --test_path , -tp     test path (default: None)

Model:
  --cell_dim , -cd      Cell dimensionality (default: 256)
  --num_layers , -nl    Number of layers (default: 1)
  --src_vocab_size , -svs
                        Vocabulary size for source language (default: 30000)
  --tgt_vocab_size , -tvs
                        Vocabulary size for target language (default: 15000)
```
