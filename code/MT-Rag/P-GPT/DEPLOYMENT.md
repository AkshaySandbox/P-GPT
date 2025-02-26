# Deploying Pregnancy GPT to Hugging Face Spaces

This guide explains how to deploy the Pregnancy GPT application to Hugging Face Spaces.

## Prerequisites

1. A Hugging Face account
2. Hugging Face CLI installed (`pip install huggingface_hub`)
3. Logged in to Hugging Face (`huggingface-cli login`)

## Deployment Steps

### 1. Using the Deployment Script

The easiest way to deploy is using the provided deployment script:

```bash
# Deploy to default space (AkshaySandbox/pregnancy-gpt)
./deploy_to_hf.py --readme --clean-data

# Deploy to a custom space
./deploy_to_hf.py YourUsername/your-space-name --readme --clean-data
```

Options:
- `--readme`: Use the Hugging Face specific README (README-HF.md)
- `--clean-data`: Remove large data files before pushing
- `--force`: Force overwrite of existing files

### 2. Manual Deployment

If you prefer to deploy manually:

1. Create a Space on Hugging Face:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Select "Docker" as the SDK
   - Enter a name for your Space
   - Click "Create Space"

2. Push your code to the Space:
   ```bash
   huggingface-cli upload YourUsername/your-space-name . --repo-type space
   ```

### 3. Setting Environment Variables

After deployment, you need to set the following environment variables in your Space:

1. Go to your Space settings
2. Add the following variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `TAVILY_API_KEY`: Your Tavily API key
   - `HF_TOKEN`: Your Hugging Face token (if needed)

## Troubleshooting

### Common Issues

1. **Uploading to a model repository instead of a Space**
   - Make sure to use `--repo-type space` when uploading
   - Check that you're using the correct Space name format (username/space-name)

2. **Large files causing upload issues**
   - Use the `--clean-data` option to remove large data files
   - Consider using Git LFS for large files

3. **Docker build failures**
   - Check the build logs in the Hugging Face Space
   - Ensure all dependencies are correctly specified in requirements.txt

4. **Configuration error: Missing configuration in README**
   - Hugging Face Spaces requires specific configuration metadata at the top of the README.md file
   - Always use the `--readme` flag when deploying with the script to ensure proper configuration
   - If deploying manually, make sure your README.md starts with:
     ```
     ---
     title: Pregnancy GPT
     emoji: 👶
     colorFrom: pink
     colorTo: purple
     sdk: docker
     app_file: app.py
     app_port: 7860
     pinned: true
     fullWidth: true
     ---
     ```

### Getting Help

If you encounter issues, check:
- Hugging Face Spaces documentation: https://huggingface.co/docs/hub/spaces
- Hugging Face CLI documentation: https://huggingface.co/docs/huggingface_hub/main/en/guides/cli 