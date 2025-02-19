# When Evolution Strategy Meets Language Models Tuning (COLING 2025)

## Acknowledgement
Our code is based on the codes of ICML2024 [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://arxiv.org/abs/2402.03898) and ICLR2024 [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/pdf/2306.08543.pdf) 

## Environment
Create a Python virtual environment and install required libraries
```bash
conda create -n eso python=3.11 && conda activate eso
pip install -r requirements.txt
```

### Data Processing
Follow the code of ICML2024 [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://arxiv.org/abs/2402.03898) to perform data processing

## Train
```bash
bash scripts/gpt2/eso/run.sh
```

## Evaluate
```bash
bash scripts/eval/eval.sh
```

## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{huang-eso-2025,
    title = "When Evolution Strategy Meets Language Models Tuning",
    author = "Huang, Bo  and
      Jiang, Yuxin  and
      Chen, Mingyang  and
      Wang, Yi  and
      Chen, Hongyang  and
      Wang, Wei",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.357/",
    pages = "5333--5344",
}
```

