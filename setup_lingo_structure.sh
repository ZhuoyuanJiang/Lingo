#!/bin/bash

# Create root level files (if they don't exist)
touch README.md
touch requirements.txt
touch setup.py
touch .gitignore

# Create data directory structure
mkdir -p data/raw/{text,audio,video,multimodal}
mkdir -p data/processed/{text,audio,video,multimodal}
mkdir -p data/metadata

# Create notebooks directory structure
mkdir -p notebooks/exploration
touch notebooks/exploration/{text_eda,audio_eda,video_eda,multimodal_eda}.ipynb

mkdir -p notebooks/preprocessing
touch notebooks/preprocessing/{text_preprocessing,audio_preprocessing,video_preprocessing}.ipynb

mkdir -p notebooks/modeling
touch notebooks/modeling/{text_model,audio_model,video_model,bimodal_fusion,trimodal_fusion}.ipynb

mkdir -p notebooks/evaluation
touch notebooks/evaluation/{baseline_evaluation,ablation_studies,error_analysis}.ipynb

# Create src directory structure
mkdir -p src/{data,models,training,utils,inference}
touch src/__init__.py

# Create data module
touch src/data/__init__.py
touch src/data/{data_loader,text_processor,audio_processor,video_processor,augmentation}.py

# Create models module
touch src/models/__init__.py
touch src/models/{text_model,audio_model,video_model,fusion_models,attention}.py

# Create training module
touch src/training/__init__.py
touch src/training/{trainer,loss_functions,metrics}.py

# Create utils module
touch src/utils/__init__.py
touch src/utils/{config,logger,visualization,io_utils}.py

# Create inference module
touch src/inference/__init__.py
touch src/inference/{predictor,postprocessing}.py

# Create configs directory
mkdir -p configs
touch configs/{data_config,model_config,training_config}.yaml

# Create scripts directory
mkdir -p scripts
touch scripts/download_datasets.sh
touch scripts/{preprocess_data,train_model,evaluate_model,run_inference}.py

# Create models directory for checkpoints
mkdir -p models/{text_models,audio_models,video_models,bimodal_models,trimodal_models}

# Create results directory
mkdir -p results/{metrics,visualizations,attention_maps,ablation_studies}

# Create chrome extension directory
mkdir -p chrome_extension/{popup,content,background,assets,lib}
touch chrome_extension/manifest.json

# Create docs directory
mkdir -p docs/presentations
touch docs/{project_report,api_documentation,user_guide}.md

echo "LINGO project directory structure created successfully!"
