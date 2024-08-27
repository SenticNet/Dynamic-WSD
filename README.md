# sentiment-analysis-with-WSD

This repository contains the multitask learning model proposed in [Neurosymbolic sentiment analysis with dynamic word sense disambiguation](https://aclanthology.org/2023.findings-emnlp.587.pdf). 



## Usage
To pretrain and test the lexical substitution model, put the desired pretrained language model in the `alm_path`, and download the required datasets to the corresponding folders in the `data` folder. Then run the following example script:
```
python pretrain.py --batch_size 20 --lr 1e-8 --alm_path "./ckpt/saved_ckpt/ALM.pt"
```
To train and test the sentiment analysis model, put the path of the trained lexical substitution model in the `lex_path`. Then run the following example script:
```
python run.py --batch_size 10 --lr 1e-6 --lex_path "./ckpt/saved_ckpt/lex_sub.pt"
```

## Citation
If you use this knowledge base in your work, please cite the paper - [Neurosymbolic sentiment analysis with dynamic word sense disambiguation](https://aclanthology.org/2023.findings-emnlp.587.pdf) with the following:
```
@inproceedings{zhang-etal-2023-neuro,
    title = "Neuro-Symbolic Sentiment Analysis with Dynamic Word Sense Disambiguation",
    author = "Zhang, Xulang  and
      Mao, Rui  and
      He, Kai  and
      Cambria, Erik",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.587",
    doi = "10.18653/v1/2023.findings-emnlp.587",
    pages = "8772--8783",
}
```
