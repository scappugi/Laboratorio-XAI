"""
build_multiclass_dataset.py
Costruisce train/test.csv con una singola etichetta a 9 vie:
  0-7 = categoria dominante (la prima tra quelle vere, in ordine di indice)
  8   = null (nessuna categoria unfair presente)
  Nota sulla scelta min(lbls): quando una clausola ha più categorie
  vere (es. [0, 3]), qui scegli sempre la prima/più bassa come "dominante"
  è una semplificazione arbitraria ma semplice da giustificare nel report 
  ("in caso di clausola multi-categoria, si è scelta la categoria di indice più basso come rappresentativa,
  per ridurre il problema a singola etichetta").
"""
import os
import pandas as pd
from datasets import load_dataset

NUM_CATEGORIES = 8
NULL_LABEL = 8  # 9a classe


def build_split(hf_split):
    texts = hf_split['text']
    multilabels = hf_split['labels']

    dominant_labels = []
    for lbls in multilabels:
        if len(lbls) == 0:
            dominant_labels.append(NULL_LABEL)
        else:
            dominant_labels.append(min(lbls))  # scelta: la categoria di indice più basso tra quelle vere

    return pd.DataFrame({'review': texts, 'sentiment': dominant_labels})


def main(base_folder, data_set_name='UnfairToS_MultiClass'):
    dataset = load_dataset("coastalcph/lex_glue", "unfair_tos")

    out_dir = os.path.join(base_folder, data_set_name)
    os.makedirs(out_dir, exist_ok=True)

    train_df = build_split(dataset['train'])
    test_df = build_split(dataset['test'])
    val_df = build_split(dataset['validation'])

    train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'validation_holdout.csv'), index=False)

    print(train_df['sentiment'].value_counts().sort_index())
    print(f"File salvati in: {out_dir}")

    return out_dir


if __name__ == '__main__':
    main(base_folder='./Datasets_LexGlue')