# Slang Detection Text-Only Model

This repository contains everything you need to train and evaluate our text-only slang detection model.

---

## Prerequisites: git & conda & jupyter notebook 



## Running Instructions: 

1. Clone the repository

2. Environment Setup: Set up the environment using environment.yml file (all dependencies and external libraries along with their versions are included): 

        # From inside the repo root
        conda env create -f environment.yml
        conda activate musicClaGen_env


3. To get the raw data, go to: https://github.com/amazon-science/slang-llm-benchmark?tab=readme-ov-file and click the "legacy version" button. This will automatically download the en.zip file. Put the en.zip folder inside the raw/text/ folder. 

4. After you have downloaded the en.zip file, you can unzip it and drag/put the "en" folder inside the data/raw/text/ folder. so your directory should look like  data/raw/text/en/ OR you can put en.zip inside the data/raw/text/ folder and unzip it there. 

5. Then run the data/raw/text/reconstruct_data.py file to reconstruct the dataset. You can also look at the lookatdata.ipynb to see what the dataset should look like. 

6. Go to data/processed/text/ and run the dataset_preparation.ipynb file to preprocess the dataset. After this, you will get 4 json files: train.json, test.json, val.json, and sample.json where sample.json is a subset of train.json. Specifically, the train.json file is the processed dataset that we will use for fine-tuning the text-only model. 

7. Train the text-only model: After you have the processed dataset, go to the /scripts folder and run the train_text_model2.ipynb file to train the text-only model. This file is the entire training pipeline for the text-only model. The path is scripts/train_text_model2.ipynb. 

8.Evaluate the model: After the training of text-only model is done, you can run scripts/model_evaluation.ipynb to evaluate the model. 


Note: This directory structure is complicated because in the proposal we want to in the future scale up to multimodality models and we have created folders for building relevant slang detection app. Therefore, the relavant parts for the text-only model are only the files that I mentioned above


        1. data/raw/text/en.zip 
        2. data\raw\text\reconstruct_data.py
        3. data\raw\text\lookatdata.ipynb
        4. data\processed\text\dataset_preparation.ipynb
        5. scripts\train_text_model_2.ipynb
        6. scripts\model_evaluation.ipynb


At the bottom there is a directory structure for your reference.





## Model Configurations:


Model 3: fine tuned using a single RTX 4070 GPU, 8GB VRAM for ~94 hours. (model and training configurations are in /configs and are named model_config.yaml and training_config.yaml respectively)

Model Config:

    text_slang_detector:
    lora_params:
        alpha: 32
        dropout: 0.1
        r: 16
        target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
    name: Qwen/Qwen2.5-1.5B-Instruct

Training Config:

        text_slang_detector:
        batch_size: 4
        epochs: 3
        gradient_accumulation_steps: 4
        lr: 0.0002
        max_length: 512
        warmup_ratio: 0.03





## A General Directory Structure: 

(not finalized and some parts are still under construction(some files just have a bare skeleton and are not functional))

LINGO/
├── README.md                  # Project overview, setup instructions, usage guide
├── environment.yml            # Environment setup
├── setup.py                   # Package installation script
├── .gitignore                 # Git ignore file
│
├── data/
│   ├── raw/                   # Original datasets
│   │   ├── text/              # raw OpenSubtitles dataset 
│   │   ├── audio/             # Audio recordings
│   │   ├── video/             # Video content
│   │   └── multimodal/        # MELD, IEMOCAP datasets
│   │
│   ├── processed/             # Preprocessed data
│   │   ├── text/              # Processed text features
│   │   ├── audio/             # Processed audio features
│   │   ├── video/             # Processed video features
│   │   └── multimodal/        # Combined features
│   │
│   └── metadata/              # Dataset descriptions, statistics, splits
│
├── notebooks/
│   ├── exploration/           # Data exploration notebooks
│   │   ├── text_eda.ipynb
│   │   ├── audio_eda.ipynb
│   │   ├── video_eda.ipynb
│   │   └── multimodal_eda.ipynb
│   │
│   ├── preprocessing/         # Data preprocessing notebooks
│   │   ├── text_preprocessing.ipynb
│   │   ├── audio_preprocessing.ipynb
│   │   └── video_preprocessing.ipynb
│   │
│   ├── modeling/              # Model development notebooks
│   │   ├── text_model.ipynb
│   │   ├── audio_model.ipynb
│   │   ├── video_model.ipynb
│   │   ├── bimodal_fusion.ipynb
│   │   └── trimodal_fusion.ipynb
│   │
│   └── evaluation/            # Model evaluation notebooks
│       ├── baseline_evaluation.ipynb
│       ├── ablation_studies.ipynb
│       └── error_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Data loading utilities
│   │   ├── text_processor.py  # Text preprocessing
│   │   ├── audio_processor.py # Audio preprocessing
│   │   ├── video_processor.py # Video preprocessing
│   │   └── augmentation.py    # Data augmentation techniques
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_model.py      # RoBERTa-based text model
│   │   ├── audio_model.py     # Wav2Vec2.0-based audio model
│   │   ├── video_model.py     # Vision model for video processing
│   │   ├── fusion_models.py   # Multimodal fusion architectures
│   │   └── attention.py       # Custom attention mechanisms
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop implementation
│   │   ├── loss_functions.py  # Custom loss functions
│   │   └── metrics.py         # Evaluation metrics
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   ├── logger.py          # Logging utilities
│   │   ├── visualization.py   # Visualization helpers
│   │   └── io_utils.py        # I/O utilities
│   │
│   └── inference/
│       ├── __init__.py
│       ├── predictor.py       # Inference pipeline
│       └── postprocessing.py  # Output processing
│
├── configs/                   # Configuration files
│   ├── data_config.yaml       # Data processing configs
│   ├── model_config.yaml      # Model architecture configs
│   └── training_config.yaml   # Training hyperparameters
│
├── scripts/
│   ├── download_datasets.sh   # Dataset download scripts
│   ├── preprocess_data.py     # Data preprocessing script
│   ├── train_model.py         # Model training script
│   ├── evaluate_model.py      # Model evaluation script
│   └── run_inference.py       # Inference script
│
├── models/                    # Saved model checkpoints
│   ├── text_models/           # Text-only model checkpoints
│   ├── audio_models/          # Audio-only model checkpoints
│   ├── video_models/          # Video-only model checkpoints
│   ├── bimodal_models/        # Bimodal fusion model checkpoints
│   └── trimodal_models/       # Trimodal fusion model checkpoints
│
├── results/
│   ├── metrics/               # Evaluation metrics
│   ├── visualizations/        # Plots and visualizations
│   ├── attention_maps/        # Attention visualization
│   └── ablation_studies/      # Ablation study results
│
├── chrome_extension/
│   ├── manifest.json          # Extension manifest
│   ├── popup/                 # Extension popup UI
│   ├── content/               # Content scripts
│   ├── background/            # Background scripts
│   ├── assets/                # Extension assets
│   └── lib/                   # JS libraries and model integration
│
└── docs/
    ├── project_report.md      # Detailed project report
    ├── api_documentation.md   # API documentation
    ├── user_guide.md          # User guide for the extension
    └── presentations/         # Presentation materials
