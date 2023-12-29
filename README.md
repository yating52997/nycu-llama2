# nycu-llama2
## Dataset
We collect data containing documents from NYCU and some information about our department, then use GPT to generate QA pairs. Finally, complete the dataset with 3000 pairs.
The dataset can be downloaded here:
[benchang1110/NYCU_QA](https://huggingface.co/datasets/benchang1110/NYCU_QA)

## Training
```py
python main.py
```
If ypu are using V100, the training time is roughly 10 minutes per epoch.
You may want to run inference to test the model
```py
python inference.py
```
You can upload the model to huggingface hub using git-lfs.
## Deployment
### Frontend
Clone chat-ui from huggingface
```bash
git clone https://github.com/huggingface/chat-ui.git
```
add .env.local in the folder, and set the following variables
```bash
MONGODB_URL=
HF_ACCESS_TOKEN=
SERPER_API_KEY=
```
you may change your port by modifying vite.config.ts
For more information, you may refer to the original [repository](https://github.com/huggingface/chat-ui)

### Backend
Clone text-generation-inference from huggingface
```bash
git clone https://github.com/huggingface/text-generation-inference.git
```
replace the Makefile, and type in terminal:
```bash
make run-nycu-llama
```
