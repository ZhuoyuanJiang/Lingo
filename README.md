Running Instructions: 

1. Clone the repository

2. To get the raw data, go to: https://github.com/amazon-science/slang-llm-benchmark?tab=readme-ov-file and click the "legacy version" button. This will automatically download the en.zip file. Put the en.zip folder inside the raw/text/ folder. 

3. After you have downloaded the en.zip file, you can unzip it and put the en folder inside the raw/text/ folder. 

4. Then run the reconstruct_data.py file to reconstruct the dataset. You can also look at the lookatdata.ipynb to see what the dataset should look like. 

5. Go to data/processed/text/ and run the dataset_preparation.ipynb file to preprocess the dataset. After this, you will get 4 json files: train.json, test.json, val.json, and sample.json where sample.json is a subset of train.json. 

6. After you have the processed dataset, go to the /scripts folder and run the train_text_model2.ipynb file to train the text-only model. This file is the entire training pipeline for the text-only model. The path is scripts/train_text_model2.ipynb. 

7. After the training of text-only model is done, you can run scripts\model_evaluation.ipynb to evaluate the model. 

Note: This directory structure is complicated because in the proposal we want to in the future scale up to multimodality models and we have created folders for building relevant slang detection app. Therefore, the relavant parts for the text-only model are only the files that I mentioned above. 







