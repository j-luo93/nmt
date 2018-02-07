## Training

`python main.py train <data_dir> -c -s <source_lang> -cd <cell_dimension/hidden_size>` 
`-c` flag is used to enable CUDA/GPU.

## Testing
`python main.py test <data_dir> -c -s <source_lang> -cd <cell_dimension/hidden_size> -mp <model_path>` 

some useful flags:
`-bw`: beam width (for testing)
`-ru`: replace UNKs
