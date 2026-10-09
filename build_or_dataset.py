"""
build_or_dataset.py

Prepara il dataset LexGLUE unfair_tos per un training binario "OR":
    label = 1  se almeno una delle 8 categorie di clausola sleale e' presente
    label = 0  altrimenti

Salva due gruppi di file per ciascuno split (train / test / validation_holdout):
- <split>.csv            -> colonne 'review','sentiment' (0/1), compatibili
                             con utils.load_data() / get_data_loader()
- <split>_multilabel.csv -> colonne 'review', label_0 ... label_7 (0/1),
                             stesso ordine di righe di <split>.csv, usate SOLO
                             per l'analisi post-hoc prototipo -> categoria,
                             mai per il training.

IMPORTANTE sulla nomenclatura (per coerenza con get_data_loader in utils.py,
che nel progetto usa 'test.csv' come validation set durante il training):
- train.csv               <- split 'train' di HF
- test.csv                <- split 'test' di HF (usato come validation dal loop di training)
- validation_holdout.csv  <- split 'validation' di HF, tenuto da parte come
                              vero test finale, da non guardare finche' non hai finito
"""

import os
import pandas as pd
from datasets import load_dataset

NUM_CATEGORIES = 8  # categorie 0..7 di unfair_tos


def build_split(hf_split):
    texts = hf_split['text']
    multilabels = hf_split['labels']  # lista di liste di interi 0..7

    binary_labels = [1 if len(lbls) > 0 else 0 for lbls in multilabels]

    onehot = []
    for lbls in multilabels:
        row = [0] * NUM_CATEGORIES
        for l in lbls:
            row[l] = 1
        onehot.append(row)

    df_binary = pd.DataFrame({'review': texts, 'sentiment': binary_labels})
    df_multi = pd.DataFrame(onehot, columns=[f'label_{i}' for i in range(NUM_CATEGORIES)])
    df_multi.insert(0, 'review', texts)

    return df_binary, df_multi


def main(base_folder, data_set_name='UnfairToS'):
    dataset = load_dataset("coastalcph/lex_glue", "unfair_tos")
    print(dataset)  # controlla che i nomi dei campi ('text','labels') combacino

    out_dir = os.path.join(base_folder, data_set_name)
    os.makedirs(out_dir, exist_ok=True)

    train_bin, train_multi = build_split(dataset['train'])
    test_bin, test_multi = build_split(dataset['test'])
    val_bin, val_multi = build_split(dataset['validation'])

    train_bin.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    test_bin.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    val_bin.to_csv(os.path.join(out_dir, 'validation_holdout.csv'), index=False)

    train_multi.to_csv(os.path.join(out_dir, 'train_multilabel.csv'), index=False)
    test_multi.to_csv(os.path.join(out_dir, 'test_multilabel.csv'), index=False)
    val_multi.to_csv(os.path.join(out_dir, 'validation_multilabel.csv'), index=False)

    print(f"Positivi (OR) nel train: {train_bin['sentiment'].sum()} / {len(train_bin)} "
          f"({100*train_bin['sentiment'].mean():.1f}%)")
    print(f"Positivi (OR) nel test (=validation di training):  "
          f"{test_bin['sentiment'].sum()} / {len(test_bin)} "
          f"({100*test_bin['sentiment'].mean():.1f}%)")
    print(f"File salvati in: {out_dir}")

    return out_dir


if __name__ == '__main__':
    main(base_folder='./Datasets_LexGlue')
