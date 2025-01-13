# Unveiling the Potential of Text in High-Dimensional Time Series Forecasting

This paper was accepted by NeurIPS 2024 TSALM Workshop. We provide the open-source code here.
This is the official repository for "Unveiling the Potential of Text in High-Dimensional Time Series Forecasting" ***(Accepted by NeurIPS-24 TSALM Workshop)*** [[Paper]](https://openreview.net/pdf?id=6666666666) <br>

🌟 If you find this work helpful, please consider to star this repository and cite our research:
```
@inproceedings{xin2024textfusionhts,
  author = {Zhou, Xin and Wang, Weiqing and Qu, Shilin and Zhiqiang, Zhang and Bergmeir, Christoph},
  title = {Unveiling the Potential of Text in High-Dimensional Time Series Forecasting},
  year = {2024},
}
```

## Datasets
Please access the well pre-processed Wiki-People and News datasets from [[Google Drive]](https://drive.google.com/drive/folders/1GgaMDso5rEJu0Rc9XkNE9wXBqa4hm1yJ?usp=drive_link), then place the downloaded contents under the corresponding folders of `/datasets`

## Quick Demo
1. Clone this repository
```
git clone git@github.com:xinzzzhou/TextFusionHTS.git
cd TextFusionHTS
```
2. Config environment
```
conda create --name textfusionhts python=3.8
conda activate textfusionhts
pip3 install -r requirements.txt
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
3. Download datasets and place them under the corresponding folders of `/datasets`
4. Embedding learning, please note: there are required login Hugging Face, please replace the `xxxxxxxxxx` with your own Hugging Face token.
```
python emb_learning.py
```
5. Train and test the model. We provide two main.py files for demonstration purposes under the root folder. 
```
python main_wiki.py
```

## Tips
Drop_last will influence the number of data windows in the end. To achieve a fair comparison, we didn't use drop_last for testing. 

## Acknowledgement
1. We gratefully acknowledge the support of Google for providing a travel grant, which enabled attendance at NeurIPS 2024. This funding has significantly contributed to our research efforts and facilitated essential academic exchange. We sincerely appreciate Google's generous support.
2. Our implementation adapts [Time-Series-Library](https://github.com/thuml/Time-Series-Library) as the code base and has extensively modified it to our purposes. We thank the authors for sharing their implementations and related resources.
