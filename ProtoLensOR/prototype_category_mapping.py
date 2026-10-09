"""
prototype_category_mapping.py

Analisi post-hoc per la modalita' "OR" a singolo classificatore:
- Il modello e' allenato solo con l'etichetta binaria (unfair si'/no).
- Qui, usando le etichette multilabel originali (0..7) del validation/test
  set (file *_multilabel.csv generato da build_or_dataset.py), costruiamo
  una mappa prototipo -> categoria piu' probabile, basata sulla correlazione
  tra l'attivazione di ciascun prototipo e la presenza di ciascuna categoria.
- In inferenza, per un nuovo esempio classificato "unfair", guardiamo quali
  prototipi si attivano di piu' e usiamo questa mappa per stimare a quale
  categoria appartenga la clausola rilevata.

Richiede la patch a PLens.py: forward(..., return_prototype_activations=True)
deve restituire anche il tensore `similarity` (attivazione per prototipo).
"""

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def get_prototype_activations(model, dataloader, device):
    """Ritorna (activations, texts). activations ha shape (N, num_prototypes)."""
    model.eval()
    all_activations = []
    all_texts = []
    for batch_num, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        special_tokens_mask = batch['special_tokens_mask'].to(device)
        original_text = batch['original_text']

        _, _, _, similarity = model(
            input_ids=input_ids, attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask, mode="test",
            original_text=original_text, current_batch_num=batch_num,
            return_prototype_activations=True
        )
        all_activations.append(similarity.detach().cpu().numpy())
        all_texts.extend(original_text)

    return np.concatenate(all_activations, axis=0), all_texts


def build_prototype_category_map(activations, multilabel_df, num_categories=8, min_corr=0.05):
    """
    activations   : (N, num_prototypes), stesso ordine di righe di multilabel_df
    multilabel_df : DataFrame con colonne label_0..label_{num_categories-1}
    """
    num_prototypes = activations.shape[1]
    corr_matrix = np.zeros((num_prototypes, num_categories))

    for k in range(num_prototypes):
        for c in range(num_categories):
            y = multilabel_df[f'label_{c}'].values
            if y.sum() == 0 or y.sum() == len(y):
                corr_matrix[k, c] = 0.0
                continue
            corr_matrix[k, c] = np.corrcoef(activations[:, k], y)[0, 1]

    corr_df = pd.DataFrame(
        corr_matrix,
        index=[f'prototype_{k}' for k in range(num_prototypes)],
        columns=[f'category_{c}' for c in range(num_categories)]
    )

    mapping = {}
    for k in range(num_prototypes):
        best_c = int(np.argmax(corr_matrix[k]))
        best_val = corr_matrix[k, best_c]
        mapping[k] = best_c if best_val >= min_corr else None

    return corr_df, mapping


def predict_category_for_example(model, single_batch, device, mapping, top_k=1):
    """
    Dato un batch (anche di un solo esempio) gia' classificato come "unfair",
    stima la/le categorie guardando i prototipi che contribuiscono di piu'
    al logit della classe positiva (indice 1).
    """
    model.eval()
    with torch.no_grad():
        input_ids = single_batch['input_ids'].to(device)
        attention_mask = single_batch['attention_mask'].to(device)
        special_tokens_mask = single_batch['special_tokens_mask'].to(device)
        original_text = single_batch['original_text']

        _, _, _, similarity = model(
            input_ids=input_ids, attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask, mode="test",
            original_text=original_text, current_batch_num=0,
            return_prototype_activations=True
        )

        fc_weight_positive = model.fc.weight[1].detach().cpu().numpy()  # (num_prototypes,)
        contribution = similarity.detach().cpu().numpy() * fc_weight_positive  # (batch, num_prototypes)

        results = []
        for row in contribution:
            top_protos = np.argsort(row)[::-1][:top_k]
            categories = [mapping.get(int(p)) for p in top_protos]
            results.append(list(zip(top_protos.tolist(), categories)))
        return results


if __name__ == '__main__':
    """
    Esempio d'uso (da lanciare dopo il training, con model/val_dataloader/device
    ancora in memoria, oppure ricaricando il checkpoint):

        from prototype_category_mapping import (
            get_prototype_activations, build_prototype_category_map, predict_category_for_example
        )

        activations, texts = get_prototype_activations(model, val_dataloader, device)

        multilabel_df = pd.read_csv(
            os.path.join(pnfrl_args.base_folder, pnfrl_args.data_set, 'test_multilabel.csv')
        )

        corr_df, mapping = build_prototype_category_map(activations, multilabel_df)
        print(corr_df.round(2))
        print(mapping)  # es. {0: 3, 1: None, 2: 6, ...}

        # inferenza su un nuovo batch (predetto "unfair" dal modello):
        result = predict_category_for_example(model, nuovo_batch, device, mapping, top_k=2)
        print(result)  # es. [[(7, 3), (2, 6)]] -> prototipo 7 -> categoria 3, prototipo 2 -> categoria 6
    """
    pass
