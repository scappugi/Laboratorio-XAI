from collections import Counter
from sklearn.metrics import f1_score
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BertConfig, BertModel
import os
import sys
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import torch.nn.functional as F
from torch import where, rand_like, zeros_like, log, sigmoid
from torch.nn import Module
# from settings import *
# # import torch.cuda.amp.GradScaler
# from settings import *
import numpy as np
from transformers import AutoTokenizer, BertModel
from transformers import  DistilBertModel, DistilBertTokenizerFast
import logging
from scipy.spatial.distance import cdist
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import KFold
from utils import *
# from model_ours import *

from PLens import *
import gc
#from gpu_mem_track import MemTracker
from torch.cuda.amp import autocast
import os 

import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
#from Gumbel_Sigmoid import *
from tqdm.auto import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# def evaluate(model, data_loader, device):
#     model.eval()
#     predictions = []
#     actual_labels = []
#     incorrect_indices = []
#     with torch.no_grad():
#         for batch_num,batch in enumerate(data_loader):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             special_tokens_mask = batch['special_tokens_mask'].to(device)
#             labels = batch['label'].to(device)
#             offset_mapping = batch['offset_mapping']
#             processed_text = batch['processed_text']
#             original_text = batch['original_text']
#             outputs, _, _ = model(input_ids=input_ids, attention_mask=attention_mask,special_tokens_mask=special_tokens_mask, mode="test", original_text = original_text, current_batch_num=batch_num)
#             _, preds = torch.max(outputs, dim=1)
#             predictions.extend(preds.cpu().tolist())
#             actual_labels.extend(labels.cpu().tolist())
#     for idx, (pred, actual) in enumerate(zip(predictions, actual_labels)):
#         if pred != actual:
#             incorrect_indices.append(idx)
#     logging.info(incorrect_indices)
#     return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)



def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    incorrect_indices = []
    with torch.no_grad():
        for batch_num,batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            special_tokens_mask = batch['special_tokens_mask'].to(device)
            labels = batch['label'].to(device)
            offset_mapping = batch['offset_mapping']
            processed_text = batch['processed_text']
            original_text = batch['original_text']
            outputs, _, _ = model(input_ids=input_ids, attention_mask=attention_mask,special_tokens_mask=special_tokens_mask, mode="test", original_text = original_text, current_batch_num=batch_num)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    for idx, (pred, actual) in enumerate(zip(predictions, actual_labels)):
        if pred != actual:
            incorrect_indices.append(idx)
    logging.info(incorrect_indices)
    acc = accuracy_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions, average="binary")
    report = classification_report(actual_labels, predictions)

    return acc, f1, report


def gold_eval(model, data_loader, p_sent2pid, device, epo_num):
    model.eval()
    predictions = []
    actual_labels = []
    document = docx.Document()
    # with torch.no_grad():
    for batch in data_loader:
        original_text = batch['original_text']
        input_ids = batch['input_ids'].to(device)
        input_attention_mask = batch['attention_mask'].to(device)
        special_tokens_mask = batch['special_tokens_mask'].to(device)
        label = batch['label_text']
        pid = [p_sent2pid[i] for i in label]
        attention_mask_label = batch["attention_mask_label"].to(device)
        special_tokens_mask_label = batch["special_tokens_mask_label"].to(device)
        input_ids_label = batch["input_ids_label"].to(device)
        # logging.info(special_tokens_mask_label)
        label_ids = (1 - special_tokens_mask_label) * input_ids_label
        # Initialize the label mask with zeros
        label_mask = torch.zeros_like(input_ids)
        # non_zero_elements = label_ids[label_ids != 0]
        seq = []
        for i in range(label_ids.shape[0]):
            # Get non-zero elements from label_ids for this sequence
            non_zero_elements = label_ids[i][label_ids[i] != 0]
            seq.append(non_zero_elements)
            # Check if the non-zero elements exist in input_ids
            for elem in non_zero_elements:
                # Get the index where the element appears in input_ids
                indices = torch.where(input_ids[i] == elem)[0]
                # Set these positions in the label mask to 1
                label_mask[i][indices] = 1
        # label_mask = find_and_replace_subsequence(label_mask, seq)

        # words, _, df, vocab = model.get_words(original_text)
        # label_words_in_order = []
        # for label_text in label:
        #     _, temp, _, label_vocab = model.get_words([label_text])
        #     label_words_in_order.append(temp)

        # token_embeddings, candidates_id, candidates = model.get_token_embedding(original_text, words, vocab)
        # label_mask = np.full((len(original_text), model.max_length), 0)
        # for index in range(len(original_text)):
        #     for i,_ in enumerate(candidates[index]):
        #         if i >= model.max_length:
        #             continue
        #         if candidates[index][i] in label_words_in_order[index]:
        #             label_mask[index, i] = 1
        # label_mask = torch.tensor(label_mask).cuda()
        # attention_mask = torch.tensor(np.where(candidates_id == 'None', 0, 1)).cuda() # candidates = torch.tensor(candidates)
        outputs = model.bert(input_ids=input_ids, attention_mask=input_attention_mask, output_hidden_states=True)
        input_token_emb = F.normalize(outputs[0], p=2, dim=-1) # (16, 512, 768)
        align = F.normalize(model.bert(input_ids_label, attention_mask_label).last_hidden_state, p=2, dim=-1)
        attention_logits, all_selected_token_index = model.attention_layer(input_token_emb, align, att_mask_emb=special_tokens_mask, att_mask_proto=attention_mask_label)
        mask = model.AdaptiveMask(attention_logits, all_selected_token_index)  
        selected_mask = mask * input_attention_mask.unsqueeze(-1)

        # p_vecs = model.bert.encode(label, normalize_embeddings=True, convert_to_tensor=True)
        # attention_logits = token_embeddings @ p_vecs.T
        # mask = model.AdaptiveMask(attention_logits)[torch.arange(len(original_text)), :, torch.arange(len(original_text))] #(16, 512)
        # selected_mask = (mask * attention_mask).cpu().detach().numpy().astype(int) #(mask * attention_mask).cpu().detach().numpy().astype(int)
        # selected_mask_ = mask * attention_mask
        # label_mask_ = torch.tensor(label_mask).cuda()
        # proto_acc = torch.sum(selected_mask * label_mask, dim=1) / torch.sum(label_mask, dim=1)
        # proto_acc_mean = torch.mean(proto_acc)
        # proto_recall = torch.sum(selected_mask * label_mask, dim=1) / torch.sum(selected_mask, dim=1)
        # proto_recall_mean = torch.mean(proto_recall)
        # logging.info("acc:", proto_acc_mean)
        # logging.info("recall:", proto_recall_mean)
    #     word_array = np.where(selected_mask == 1, candidates_id, "None")
    #     for i in range(len(original_text)):
    #         selected_words = []
    #         for id_ in word_array[i]:
    #             if id_ != "None":
    #                 selected_words.append(words[int(id_)])
    #         positions = find_keywords(selected_words, original_text[i])
    #         document = highlight_sections_docx(positions, original_text[i], document, label[i])
    #     # selected_sent = [
    #     #         [i for i in word_array[idx, :] if i is not None]
    #     #         for idx in range(len(original_text))
    #     #     ]
    # document.save("/home/bwei2/ProtoTextClassification/_test/" + str(epo_num) + ".docx")
    # return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def find_closet_test_sentence(model, data_loader, device):
    model.eval()
    df = pd.read_csv("/home/bwei2/ProtoTextClassification/Data/test_imdb.csv")
    test_texts = df['review'].tolist()
    test_ins_emb = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:

            original_text = batch['original_text']
            test_ins_emb.append(model.bert.encode(original_text, normalize_embeddings=True, convert_to_tensor=False))
    all_test_emb = np.stack(test_ins_emb, axis=0)
    distances = cosine_similarity(model.prototype_vectors, all_test_emb)
    closest_sentences = {}
    top_k = 5
    for proto_idx, proto_distances in enumerate(distances):
        # Get indices of the top_k most similar sentences
        top_k_indices = np.argsort(proto_distances)[-top_k:][::-1]
        # Retrieve the actual sentences
        closest_sentences[proto_idx] = [test_texts[idx] for idx in top_k_indices]
    
    return closest_sentences
    



def train_step(model, val_dataloader, data_loader, optimizer, scheduler, device, new_proto, tau=1, train_texts=None, class_weights= None):
    model.train()
    Loss = 0
    model.train_text = []
    for (batch_num,batch) in enumerate(data_loader):
        model.batch_num = batch_num
        # if batch_num % 5000 == 0 and batch_num >0:
        #     accuracy, f1, report = evaluate(model, val_dataloader, device)
        #     pnfrl_args = {'bert_model_name': model.args.bert_model_name, 'num_classes': model.args.num_classes, 
        #                 'prototype_num': model.args.prototype_num, 'batch_size': model.args.batch_size, 
        #                 'hidden_dim': model.args.hidden_dim, 'max_length': model.args.max_length}
        #     # if accuracy > best_result: 
        #     #     best_result = accuracy
        #     #     torch.save({'model_state_dict': model.state_dict(), 'pnfrl_args': pnfrl_args}, model.args.model_path)
        #     if f1 > best_result:
        #       best_result = f1
        #       torch.save({'model_state_dict': model.state_dict(), 'pnfrl_args': pnfrl_args}, model.args.model_path)
        #     logging.info(f"Batch {batch_num} \n")
        #     logging.info(f"Validation Accuracy: {accuracy:.4f}")
        #     logging.info(f"Validation F1: {f1:.4f}")
        #     logging.info(report)

        LOG=False
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        special_tokens_mask = batch['special_tokens_mask'].to(device)
        labels = batch['label'].to(device)
        offset_mapping = batch['offset_mapping'].to(device)
        processed_text = batch['processed_text']
        original_text = batch['original_text']
        outputs, loss_mu, augmented_loss  = model(input_ids=input_ids, attention_mask=attention_mask, special_tokens_mask=special_tokens_mask, 
                        new_proto=new_proto, log=LOG, tau=tau, offset_mapping=offset_mapping, processed_text=processed_text, 
                        current_batch_num=batch_num, original_text = original_text)
        #//////////*************************
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels) + 0.1 * loss_mu - 0.01 * model.diversity_loss
        #*************************************************************

        #loss = nn.CrossEntropyLoss()(outputs, labels) + 0.1 * loss_mu - 0.001 * model.diversity_loss
        Loss += loss.item()
        #logging.info(f"acc {criterion(outputs, labels).item()}: {nn.CrossEntropyLoss()(outputs, labels)}")
        logging.info(f"CE loss: {criterion(outputs, labels).item():.4f}")
        loss.backward()
        print(
            "grad MDN:",
            model.mdn.mu.weight.grad.norm().item()
        )     
        print(
            "grad prototypes:",
            model.prototype_vectors.grad.norm().item()
        )  
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
    return Loss

# def train_step(model, val_dataloader, data_loader, optimizer, scheduler, device, new_proto, tau=1, train_texts=None, class_weights= None):
#     model.train()
#     Loss = 0
#     model.train_text = []
#     best_result = 0
#     pbar = tqdm(data_loader, desc="Training", leave=True, dynamic_ncols=True)
#     for (batch_num,batch) in enumerate(pbar):
#         model.batch_num = batch_num
#         if batch_num % 5000 == 0 and batch_num >0:
#             accuracy, report = evaluate(model, val_dataloader, device)
#             pnfrl_args = {'bert_model_name': model.args.bert_model_name, 'num_classes': model.args.num_classes, 
#                         'prototype_num': model.args.prototype_num, 'batch_size': model.args.batch_size, 
#                         'hidden_dim': model.args.hidden_dim, 'max_length': model.args.max_length}
#             if accuracy > best_result: 
#                 best_result = accuracy
#                 torch.save({'model_state_dict': model.state_dict(), 'pnfrl_args': pnfrl_args}, model.args.model_path)
#             logging.info(f"Batch {batch_num} \n")
#             logging.info(f"Validation Accuracy: {accuracy:.4f}\n")
#             logging.info(report)
#         LOG=False
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         special_tokens_mask = batch['special_tokens_mask'].to(device)
#         labels = batch['label'].to(device)
#         offset_mapping = batch['offset_mapping'].to(device)
#         processed_text = batch['processed_text']
#         original_text = batch['original_text']
#         outputs, loss_mu, augmented_loss  = model(input_ids=input_ids, attention_mask=attention_mask, special_tokens_mask=special_tokens_mask, 
#                         new_proto=new_proto, log=LOG, tau=tau, offset_mapping=offset_mapping, processed_text=processed_text, 
#                         current_batch_num=batch_num, original_text = original_text)
#         #//////////*************************
#         if class_weights is not None:
#             criterion = nn.CrossEntropyLoss(weight=class_weights)
#         else:
#             criterion = nn.CrossEntropyLoss()
#         loss = criterion(outputs, labels) + 0.1 * loss_mu - 0.001 * model.diversity_loss
#         #*************************************************************

#         #loss = nn.CrossEntropyLoss()(outputs, labels) + 0.1 * loss_mu - 0.001 * model.diversity_loss
#         Loss += loss.item()
#         loss.backward()
#         grad_mdn = model.mdn.mu.weight.grad.norm().item()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
#         optimizer.step()
#         scheduler.step()

#         pbar.set_postfix({
#             "loss": f"{loss.item():.3f}",
#             "grad_mdn": f"{grad_mdn:.3f}",
#         })
#     return Loss

def gpu_mem_str():
    if not torch.cuda.is_available():
        return "GPU: n/a"
    allocated = torch.cuda.memory_allocated() / 1024**3
    return f"GPU mem: {allocated:.2f}GB"

# def load_model(path, device_id, log_file=None, distributed=True):
#     checkpoint = torch.load(path, map_location='cpu')
#     saved_args = checkpoint['pnfrl_args']
#     pnfrl = BERTClassifier(bert_model_name=saved_args['bert_model_name'], 
#                            num_classes=saved_args['num_classes'], 
#                            num_prototype=saved_args['prototype_num'], 
#                            batch_size=saved_args['batch_size'], 
#                            hidden_dim=saved_args['hidden_dim'], 
#                            max_length=saved_args['max_length'])
#     pnfrl.load_state_dict(checkpoint['pnfrl_args'])
#     return pnfrl
def load_model(path, args, tokenizer, device):
    """
    Funzione corretta per caricare il modello salvato.
    """
    print(f"Caricamento dei pesi da: {path}")
    checkpoint = torch.load(path, map_location=device)
    
    # Inizializza l'architettura vuota passando gli argomenti corretti
    model = BERTClassifier(
        args=args, 
        bert_model_name=args.bert_model_name, 
        num_classes=args.num_classes, 
        num_prototype=args.prototype_num, 
        batch_size=args.batch_size, 
        hidden_dim=args.hidden_dim, 
        max_length=args.max_length, 
        tokenizer=tokenizer
    )
    
    # CARICA I PESI CORRETTI (model_state_dict, NON pnfrl_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Modello caricato con successo!")
    return model
    
def train_model(gpu, args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:250"
    rank = args.nr * args.gpus + gpu
    torch.manual_seed(args.r_seed)
    device_id = args.device_ids[gpu]
    torch.cuda.set_device(device_id)
    log_file = args.log
    if gpu == 0:
        writer = SummaryWriter(args.folder_path)
        is_rank0 = True
    else:
        writer = None
        is_rank0 = False
    if is_rank0:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
        if log_file is None:
            logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
        else:
            logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)
    adam_epsilon = 1e-8
    num_epochs = args.epoch
    dataset = args.data_set
    train_dataloader, val_dataloader, tokenizer, train_texts = get_data_loader(dataset,args.dataset_path, args.world_size, rank, args.batch_size, args.max_length, args.bert_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_warmup_steps = 0
    num_training_steps = len(train_dataloader)*num_epochs
    #*********************************************************
    # Usiamo i tuoi conti: 4902 negativi (classe 0), 630 positivi (classe 1).
    # Il peso standard per la classe 0 è 1.0, per la classe 1 è il rapporto (4902/630 = ~7.78)
    class_weights = torch.tensor([1.0, 7.78], dtype=torch.float32).to(device)
    #***************************************************
    model = BERTClassifier(args=args, bert_model_name=args.bert_model_name, num_classes=args.num_classes, 
                           num_prototype=args.prototype_num, 
                           batch_size=args.batch_size, hidden_dim=args.hidden_dim, max_length=args.max_length, 
                           tokenizer=tokenizer).to(device)
    
    model.tokenizer = tokenizer
    model.args = args
    best_result = 0
    specific_param_left = model.AdaptiveMask.current_val_left
    specific_param_right =  model.AdaptiveMask.current_val_right
    other_params = [param for name, param in model.named_parameters() if param is not specific_param_left and param is not specific_param_right]
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=adam_epsilon, weight_decay=1e-5 )
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    new_proto = None

    history_loss = []
    history_f1 = []
    history_acc = []

    Total_Loss = []
    tau = 0.6
    best_f1 = 0.0
    patience = 5          # numero di epoche senza miglioramento
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        Loss = train_step(model,val_dataloader, train_dataloader, optimizer, scheduler, device, new_proto, tau=tau, train_texts=train_texts) #Tolto class_weights=class_weights
        
        Total_Loss.append(Loss)
        history_loss.append(Loss)

        accuracy, f1, report = evaluate(model, val_dataloader, device)
        pnfrl_args = {'bert_model_name': args.bert_model_name, 'num_classes': args.num_classes, 
                    'prototype_num': args.prototype_num, 'batch_size': args.batch_size, 
                    'hidden_dim': args.hidden_dim, 'max_length': args.max_length}
        # //////veccgio blocco/////////
        # if accuracy > best_result: 
        #     best_result = accuracy
        #     # torch.save({'model_state_dict': model.state_dict(), 'pnfrl_args': pnfrl_args}, args.model_path)
        #///////////////// nuovo blocco /////////////////////////
        
        history_acc.append(accuracy)
        history_f1.append(f1)

        nome_report_epoca = f"report_LR_{args.learning_rate}_Win_{args.window_size}_Epoch_{epoch + 1}.txt"
        percorso_report_epoca = os.path.join(args.folder_path, nome_report_epoca)

        with open(percorso_report_epoca, "w") as f:
            f.write(f"=== Report Epoca {epoch + 1}/{num_epochs} ===\n")
            f.write(f"Learning Rate: {args.learning_rate} | Window Size: {args.window_size}\n")
            f.write(f"Validation Accuracy: {accuracy:.4f}\n")
            f.write(f"Validation F1: {f1:.4f}\n\n")
            f.write(report)

        if f1 > best_f1:
          best_f1 = f1
          epochs_no_improve = 0

          # vecchio modo per salvare
          # torch.save({
          #     'model_state_dict': model.state_dict(),
          #     'pnfrl_args': pnfrl_args
          # }, args.model_path)

          drive_path = '/home/inginf2026simcap/ProtoLensOR/Saved_Models/UnfairToS_Best_Model/Tuning_Results'
          os.makedirs(drive_path, exist_ok=True)
          
          torch.save({
              'model_state_dict': model.state_dict(),
              'pnfrl_args': pnfrl_args
          }, os.path.join(drive_path, 'model.pth'))

        else:
          epochs_no_improve += 1
          #////////////////////////////////
        logging.info(f"Epoch {epoch + 1}/{num_epochs} \n")
        logging.info(f"Validation Accuracy: {accuracy:.4f}\n")
        logging.info(f"Validation F1: {f1:.4f}")
        logging.info(report)
        logging.info(Total_Loss)
        if epochs_no_improve >= patience:
          logging.info(f"Early stopping dopo {epoch+1} epoche")
          break

    return model, val_dataloader, best_f1, history_loss, history_f1, history_acc



def train_main(args):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    # mp.spawn(train_model, nprocs=args.gpus, args=(args,),join=True)
    return train_model(0, args)


if __name__ == '__main__':
    from args import parser
    
    # parse_known_args() ignora gli argomenti che non conosce (come quelli di Jupyter)
    args, unknown = parser.parse_known_args()
    
    # Stampa i parametri per verifica
    print("Parametri caricati:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
        
    # Avvia l'esperimento
    train_main(args)