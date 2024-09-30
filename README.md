# Instructions for Working with the Training Dataset

## Set Up
1. Upload the train dataset to the `datasets` folder in CSV format and add the filename to `setup.py` assigned to the constant `TRAIN_FILE`.
2. Update the `CONTENT_COL` and `LABEL_COL` constants in the `setup.py` file with the corresponding column names.
3. Create a `.env` file and add your API keys to it under the names `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, and/or `META_API_KEY`.

   - If you are using version control on your fork of this repo, **make sure gitignore this `.env` file!** Do not push your API keys.
4. Run `pip install -r requirements.txt` to ensure all depedencies are installed.

## Playing around
To run a prompt-model submission against the train dataset, update the `prompt` and `model` variables in `train_submissions.py` (being sure to follow the formatting guidelines outlined in the comments) and run the module.

## Troubleshooting
- `train_submissions.py` does not store your accuracy score anywhere. Be sure to keep track of models and prompts that perform well separately or modify this script to save these results somewhere.
