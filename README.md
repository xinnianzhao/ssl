# Speech Encoder Fine-tuning for Dutch ASR

This repository contains training scripts for fine-tuning various speech encoders on Dutch ASR using the CGN dataset.

## Supported Models

- **Whisper**: OpenAI's Whisper-medium model (encoder-decoder architecture)
- **Wav2Vec2.0**: Facebook's XLSR-53 multilingual model with CTC
- **HuBERT**: Facebook's mHuBERT multilingual model with CTC
- **WavLM**: Microsoft's WavLM-MS-40k multilingual model with CTC

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

The training scripts expect data in the following format:
- Text file with one line per sample: `audio_path|transcript`
- Example: `/path/to/audio.wav|dit is een voorbeeld zin`

Create the following files:
- `cgn_train.txt`: Training data
- `cgn_eval.txt`: Validation data (optional)
- `cgn_test.txt`: Test data (optional)

## Training Pipeline

### Step 1: Train BPE Tokenizer (for CTC models)

For Wav2Vec2, HuBERT, and WavLM models, first train a BPE tokenizer:

```bash
python train_tokenizer.py \
  --input_file cgn_train.txt \
  --model_prefix ./tokenizer/dutch_bpe \
  --vocab_size 5000 \
  --model_type bpe
```

### Step 2: Fine-tune Speech Encoders

#### Whisper

```bash
python run_whisper.py \
  --model_name_or_path openai/whisper-medium \
  --train_data_path cgn_train.txt \
  --eval_data_path cgn_eval.txt \
  --output_dir ./outputs/whisper \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --num_train_epochs 10 \
  --fp16 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_steps 1000 \
  --logging_steps 100 \
  --predict_with_generate \
  --generation_max_length 225 \
  --generation_num_beams 1 \
  --language nl \
  --task transcribe \
  --do_train \
  --do_eval \
  --report_to wandb
```

#### Wav2Vec2

```bash
python run_wav2vec2.py \
  --model_name_or_path facebook/wav2vec2-xlsr-53 \
  --tokenizer_path ./tokenizer/dutch_bpe.model \
  --train_data_path cgn_train.txt \
  --eval_data_path cgn_eval.txt \
  --output_dir ./outputs/wav2vec2 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --num_train_epochs 10 \
  --fp16 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_steps 1000 \
  --logging_steps 100 \
  --gradient_checkpointing \
  --do_train \
  --do_eval \
  --report_to wandb
```

#### HuBERT

```bash
python run_hubert.py \
  --model_name_or_path facebook/hubert-xlsr-53 \
  --tokenizer_path ./tokenizer/dutch_bpe.model \
  --train_data_path cgn_train.txt \
  --eval_data_path cgn_eval.txt \
  --output_dir ./outputs/hubert \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --num_train_epochs 10 \
  --fp16 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_steps 1000 \
  --logging_steps 100 \
  --gradient_checkpointing \
  --do_train \
  --do_eval \
  --report_to wandb
```

#### WavLM

```bash
python run_wavlm.py \
  --model_name_or_path microsoft/wavlm-base-plus \
  --tokenizer_path ./tokenizer/dutch_bpe.model \
  --train_data_path cgn_train.txt \
  --eval_data_path cgn_eval.txt \
  --output_dir ./outputs/wavlm \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --num_train_epochs 10 \
  --fp16 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_steps 1000 \
  --logging_steps 100 \
  --gradient_checkpointing \
  --do_train \
  --do_eval \
  --report_to wandb
```

## Training on Server/Cluster

For submitting jobs to a server or cluster, you can create shell scripts:

### Example SLURM script (train_whisper.sh):

```bash
#!/bin/bash
#SBATCH --job-name=whisper-dutch
#SBATCH --output=logs/whisper-%j.out
#SBATCH --error=logs/whisper-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

module load cuda/11.7
module load python/3.9

source venv/bin/activate

python run_whisper.py \
  --model_name_or_path openai/whisper-medium \
  --train_data_path /data/cgn_train.txt \
  --eval_data_path /data/cgn_eval.txt \
  --output_dir /outputs/whisper \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --fp16 \
  --do_train \
  --do_eval
```

Submit with: `sbatch train_whisper.sh`

## Multi-GPU Training

For multi-GPU training, use the `accelerate` library:

```bash
accelerate config  # Configure multi-GPU settings

accelerate launch run_wav2vec2.py \
  --model_name_or_path facebook/wav2vec2-xlsr-53 \
  --tokenizer_path ./tokenizer/dutch_bpe.model \
  --train_data_path cgn_train.txt \
  --output_dir ./outputs/wav2vec2 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 10 \
  --do_train
```

## Model Outputs

After training, each model directory will contain:
- `pytorch_model.bin`: Model weights
- `config.json`: Model configuration
- `preprocessor_config.json`: Processor configuration
- `vocab.json`: Vocabulary (for CTC models)
- `tokenizer_config.json`: Tokenizer configuration
- Training metrics and logs

## Evaluation

To evaluate a trained model:

```bash
python run_whisper.py \
  --model_name_or_path ./outputs/whisper/checkpoint-5000 \
  --test_data_path cgn_test.txt \
  --output_dir ./outputs/whisper/eval \
  --per_device_eval_batch_size 16 \
  --do_predict \
  --predict_with_generate \
  --generation_max_length 225 \
  --generation_num_beams 5
```

## Tips for Training

1. **Memory Management**: 
   - Use gradient checkpointing for large models
   - Reduce batch size if OOM
   - Use fp16 mixed precision training

2. **Performance**:
   - Adjust `max_duration_in_seconds` based on your GPU memory
   - Use multiple GPUs with DDP for faster training
   - Cache preprocessed datasets for faster loading

3. **Hyperparameters**:
   - Start with the default learning rates provided
   - Monitor WER/CER metrics during training
   - Use early stopping to prevent overfitting

## WandB Integration

All scripts support WandB logging. Set your API key:

```bash
wandb login
```

Then add `--report_to wandb` to any training command.

## License

This project is for research purposes. Make sure you have the appropriate licenses for the CGN dataset and pretrained models.