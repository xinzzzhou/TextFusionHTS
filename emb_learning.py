import os
import pandas as pd
import argparse
import torch
import numpy as np 
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# login Hugging Face
login("xxxxxxxxxx")
model_id = "meta-llama/Meta-Llama-3.1-8B"

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parser():
    parser = argparse.ArgumentParser(description='text embedding learning')
    parser.add_argument('--root_path', type=str, default=os.getcwd(), help='root path')
    parser.add_argument('--txtfile_path', type=str, default=os.getcwd()+'/datasets/News/xx.csv', help='file path')
    parser.add_argument('--tsfile_path', type=str, default=os.getcwd()+'/datasets/News/xx.csv', help='ts file path')
    return parser.parse_args()

    
def align_order(ts_file, df, part=''):
    df_ts = pd.read_csv(ts_file)
    print(df_ts.shape)
    list_column = []
    for col in df_ts.columns[1:]:
        list_column.append('_'.join(col.split('_')[:-3]))
    new_txt_emb_list = []
    for lc in list_column:
        emb = df[df['id'] == lc]['embs'].values
        # transfer string to list
        emb = eval(emb[0])[0]
        new_txt_emb_list.append(emb)
    np.savez(f'txt_{part}_emb.npz', new_txt_emb_list)
    np.save(f'txt_{part}_emb.h5', new_txt_emb_list)


args = parser()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")
    
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
llama = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use half precision
    device_map="auto",  # Automatically manage model placement
    low_cpu_mem_usage=True
)

# load the data
df = pd.read_csv(args.txtfile_path)
df1 = df.copy()
df2 = df.copy()
df3 = df.copy()
text = list(df['text'].values)
avg_embs, cls_embs, bos_embs, eos_embs = [], [], [], []
# encode the text
count_total = len(text)
inputs_len = []
t_len = []

with torch.no_grad():
    for i, t in enumerate(text):
        inputs = tokenizer(t, return_tensors="pt", max_length=512, padding=True, truncation=True).to(device)
        inputs_len.append(inputs["input_ids"].shape[1])
        t_len.append(len(t))

        txt_enc = llama(**inputs, output_hidden_states=True)  # Assuming the last hidden state is the text embedding
        # get the avg embedding
        last_hidden_states = txt_enc.hidden_states[-1]
        avg_representation = last_hidden_states.mean(dim=1)
        # get the cls embedding
        cls_representation = last_hidden_states[:, 0, :]
        # get bos embedding
        bos_representation = last_hidden_states[:, 1, :]
        # get eos embedding
        eos_representation = last_hidden_states[:, -1, :]
        avg_embs.append(avg_representation.cpu().detach().numpy())
        cls_embs.append(cls_representation.cpu().detach().numpy())
        bos_embs.append(bos_representation.cpu().detach().numpy())
        eos_embs.append(eos_representation.cpu().detach().numpy())
        
        torch.cuda.empty_cache()

# save the embeddings
df['embs'] = avg_embs
df1['embs'] = cls_embs
df2['embs'] = bos_embs
df3['embs'] = eos_embs
df['embs'] = df['embs'].apply(lambda x: str(x.tolist()))
df1['embs'] = df1['embs'].apply(lambda x: str(x.tolist()))
df2['embs'] = df2['embs'].apply(lambda x: str(x.tolist()))
df3['embs'] = df3['embs'].apply(lambda x: str(x.tolist()))

# align the order of the text and the text embedding
align_order(args.tsfile_path, df, 'avg')
align_order(args.tsfile_path, df1, 'cls')
align_order(args.tsfile_path, df2, 'bos')
align_order(args.tsfile_path, df3, 'eos')



print('done!')
