# PriceLens: Qwen2-VL Product Price Prediction

Fine-tuned Qwen2-VL VLM with QLoRA to predict Amazon product prices from catalog text and product images
PriceLens is a multimodal product-price prediction workflow implemented in [`finetuningg/finetuner copy.ipynb`](finetuningg/finetuner%20copy.ipynb). It fine-tunes `unsloth/Qwen2-VL-7B-Instruct-bnb-4bit` with LoRA using product catalog text and product images from the Amazon ML Challenge 2025 dataset.

## What the notebook does

The notebook runs the following pipeline:

1. Installs the PyTorch, Unsloth, Transformers, and TRL dependencies.
2. Downloads the dataset with KaggleHub.
3. Loads `train.csv` from the downloaded dataset.
4. Downloads product images concurrently into `finetuningg/images/`.
5. Removes rows whose images could not be downloaded.
6. Creates an 80/20 train/test split.
7. Converts each row into a Qwen2-VL conversation containing an image, catalog text, and numeric price answer.
8. Loads the quantized Qwen2-VL model and attaches language-layer LoRA adapters.
9. Fine-tunes with `SFTTrainer` and `UnslothVisionDataCollator`.
10. Generates prices for the test split and calculates SMAPE.
11. Saves the LoRA adapter, tokenizer, and prediction CSV.

## Requirements

- Linux or another environment supported by the installed CUDA/PyTorch build
- CUDA-enabled GPU; the notebook stops if CUDA is unavailable
- Enough GPU memory for Qwen2-VL 7B fine-tuning. The notebook is configured for an 80 GB GPU and uses 4-bit loading plus LoRA.
- KaggleHub access to download `raghavdharwal/amazon-ml-challenge-2025`
- A working Python environment with the packages imported by the notebook:
  - `torch`, `torchvision`, `torchaudio`, and optionally `xformers`
  - `unsloth`
  - `transformers`
  - `trl`
  - `pandas`, `numpy`, `scikit-learn`, `Pillow`, `requests`, and `tqdm`
  - `kagglehub`

The first notebook cells contain commented installation commands. Adjust the PyTorch index URL and package versions to match the CUDA driver on the target machine. The notebook currently references Transformers 5.5.0 and TRL 0.22.2.

## Dataset expectations

The downloaded training CSV must contain these columns:

| Column | Purpose |
| --- | --- |
| `sample_id` | Stable identifier used to name local image files |
| `image_link` | Product image URL |
| `price` | Numeric training target in USD |
| `Content_catalougr` or `catalog_content` | Product catalog text |

The notebook searches recursively for `train.csv` under the KaggleHub download directory. Images are written to a local `images/` directory relative to the notebook's current working directory. Existing non-empty image files are reused.

Image downloads use up to 64 worker threads and retry each URL three times. Rows without a successfully indexed local image are omitted before splitting and again before conversation preparation.

## Running the notebook

Open [`finetuningg/finetuner copy.ipynb`](finetuningg/finetuner%20copy.ipynb) in VS Code or Jupyter and run the cells in order.

Recommended sequence:

1. Enable the intended Python kernel and confirm it can import PyTorch and Unsloth.
2. Run the installation cells only when the environment needs setup.
3. Run the imports and CUDA check.
4. Download the dataset and inspect the loaded training shape.
5. Download and index images.
6. Create the train/test split and multimodal conversations.
7. Load Qwen2-VL and attach LoRA adapters.
8. Construct the trainer and run fine-tuning.
9. Run the testing cell to generate predictions and save artifacts.

A fresh notebook kernel is recommended after installing or upgrading PyTorch, Transformers, TRL, or Unsloth.

## Model configuration

The model-loading cell uses:

| Setting | Value |
| --- | --- |
| Base model | `unsloth/Qwen2-VL-7B-Instruct-bnb-4bit` |
| Quantization | 4-bit loading |
| Vision layers | Frozen |
| Language layers | Fine-tuned through LoRA |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Random seed | 42 |
| Maximum image tokens | 256 |
| Maximum text tokens | 512 |
| Maximum sequence length | 1024 |
| Epochs | 1 |
| Learning rate | `2e-4` |
| Optimizer | `adamw_8bit` |
| Effective batch size | 8 |
| Data-loader workers | 2 |

The training cell chooses the per-device batch size from the configured GPU memory threshold, then derives gradient accumulation steps to maintain an effective batch size of 8. It enables BF16 when supported and otherwise uses FP16.

## Prompt and target

Each training example asks the model to predict a positive numeric price from the product image and catalog text. The intended response is one number without a currency symbol or explanation.

The instruction is equivalent to:

```text
You are a product pricing assistant. Predict the product price in USD using the catalog text and product image.
Do not treat weights, volumes, quantities, years, or model numbers as the price.
Return only one positive numeric price, without currency symbols or explanation.
```

Catalog text is truncated to 512 tokenizer tokens before training. The assistant target is formatted to two decimal places.

## Evaluation

For each test row, the notebook:

1. Generates at most 16 new tokens.
2. Extracts the first positive numeric value from the generated text.
3. Replaces invalid or missing predictions with the median training price.
4. Clips predictions to a minimum of `0.01`.
5. Reports symmetric mean absolute percentage error (SMAPE).

The evaluation cell writes:

- `qwen2vl_price_lora/`: LoRA adapter and tokenizer
- `qwen2vl_test_predictions.csv`: sample IDs, actual prices, raw model output, and predicted prices

## Important notebook notes

- The notebook downloads a Kaggle dataset at runtime; the dataset is not stored in this repository.
- `IMAGE_FOLDER` is based on `Path.cwd()`. Start the notebook from the intended project directory so images are stored where expected.
- The model-loading cell computes actual GPU memory, but the fine-tuning cell later assigns `GPU_MEMORY_GB = 80` manually. Change that line if the machine has a different GPU or use the measured value consistently.
- The fine-tuning cell redefines `convert_row_to_conversation` but currently stops after creating the user message. The earlier definition, which appends the assistant price and returns `{"messages": messages}`, should remain in effect or be restored before relying on that redefinition.
- If a run is interrupted, stop the notebook kernel and check for orphaned processes before starting again:

```bash
nvidia-smi
kill <python-pid>
# Use kill -9 <python-pid> only if the normal signal does not work.
```

## Troubleshooting

### CUDA is unavailable

The model-loading cell raises an error when `torch.cuda.is_available()` is false. Select a CUDA-enabled kernel and verify the PyTorch installation matches the installed NVIDIA driver.

### Out-of-memory errors

Reduce `PER_DEVICE_TRAIN_BATCH_SIZE`, increase `GRADIENT_ACCUMULATION_STEPS` to preserve the effective batch size, reduce `MAX_SEQ_LENGTH`, reduce `MAX_TEXT_TOKENS`, or lower `MAX_IMAGE_TOKENS`.

### Missing images

Inspect the failed-download table printed by the image cell. Check the URLs, network access, and the local `images/` directory. Re-running the image cell reuses valid files and retries missing or empty files.

### Trainer API mismatch

The notebook uses Unsloth, Transformers, and TRL APIs that are version-sensitive. Confirm the installed versions, restart the kernel after package changes, and keep the `processing_class`, `dataset_kwargs`, and vision data-collator settings aligned with the installed versions.

## Outputs and cleanup

Large runtime artifacts are created outside the source notebook, including downloaded images, model checkpoints, compiled caches, and adapter files. Keep these out of version control unless they are intentionally being distributed.

To reclaim GPU memory after a completed run, shut down the notebook kernel. For an interrupted run, identify and terminate only the Python process belonging to that notebook, then verify with `nvidia-smi` that no compute process remains.
