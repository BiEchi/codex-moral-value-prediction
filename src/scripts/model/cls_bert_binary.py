#!/usr/bin/env python
# coding: utf-8

# # 1. Data Processing

# ## 1.1 Data loading

# In[12]:


# load the data
import time
import json
import torch
import pickle
import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
from scipy.stats import entropy
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

base_address = '../../../'

for task_type in ['twitter', 'reddit']:
    for model_name in ['bert-base-uncased', 'roberta-base', 'albert-base-v2']:

        train_data = pd.read_json(base_address+f"data/moral/interim/train_test/{task_type}/train_bert_binary.json", lines=True)
        valid_data = pd.read_json(base_address+f"data/moral/interim/train_test/{task_type}/valid_bert_binary.json", lines=True)
        test_data = pd.read_json(base_address+f"data/moral/interim/train_test/{task_type}/test_bert_binary.json", lines=True)

        print('train data size: ', train_data.shape[0])
        print('valid data size: ', valid_data.shape[0])
        print('test data size: ', test_data.shape[0])


        # In[13]:


        # concat the three datasets
        data = pd.concat([train_data, valid_data, test_data], axis=0)

        train_size = train_data.shape[0]
        valid_size = valid_data.shape[0]
        test_size = test_data.shape[0]
        size = train_size + valid_size + test_size
        print("data size: ", size)

        print("Running sanity checks...")
        check1 = all(valid_data.iloc[5] == data.iloc[train_size + 5])
        check2 = all(test_data.iloc[5] == data.iloc[train_size + valid_size + 5])
        check3 = all(data[train_size + valid_size:] == test_data)
        assert check1 and check2 and check3, "The data is not concatenated correctly."

        labels = data['label'].tolist()
        unique_labels = list(set(labels))
        print("The unique labels are:", unique_labels, sep='\n')

        category_count = len(unique_labels)
        print("The number of categories is:", category_count)

        # get an index for each item
        data['index'] = data.index


        # In[14]:


        # import the BERT model
        from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

        # * sort values before extracting the input_ids, attention_masks and random_split
        sentences = data['text'].values
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        max_len = 0
        for sent in tqdm(sentences):
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens and examine the length
            input_ids = tokenizer.encode(sent, add_special_tokens=True)
            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))
            
        # if max_len > 512, then set max_len to 512 in tokenizer.encode_plus
        print("The max length is:", max_len)
        if max_len > 512:
            print("The max length is larger than 512, please set max_len to 512 in the next cell")
        if max_len < 512:
            print("The max length is smaller than 512, please input the max_len in the next cell")


        # In[15]:


        # Tokenize all of the sentences and map the tokens to their word IDs.
        input_ids = []
        attention_masks = []

        for sent in tqdm(sentences):
            encoded_dict = tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 512,           # Pad & truncate all sentences.
                                truncation=True,
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                        )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            
        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        print("size of input_ids:", input_ids.shape)
        print("size of attention_masks:", attention_masks.shape)


        # In[19]:


        def category_mapping(samples):
            """
            # create the category mapping
            """
            category_map = {}
            for i, category in enumerate(samples['label'].unique()):
                category_map[category] = i
            category_num = i+1
            # apply the mapping
            samples['label_id'] = samples['label'].apply(lambda x: category_map[x])
            return samples, category_map


        # In[20]:


        from torch.utils.data import TensorDataset, random_split, Subset

        # split the data samples
        data, category_map = category_mapping(data)
        labels = data['label_id'].values
        labels = torch.tensor(labels)
        print("category mapping is:", category_map, sep='\n')


        # In[21]:


        train_inputids = input_ids[:train_size]
        valid_inputids = input_ids[train_size:(train_size+valid_size)]
        test_inputids = input_ids[train_size+valid_size:]

        train_attention_masks = attention_masks[:train_size]
        valid_attention_masks = attention_masks[train_size:(train_size+valid_size)]
        test_attention_masks = attention_masks[train_size+valid_size:]

        train_labels = labels[:train_size]
        valid_labels = labels[train_size:(train_size+valid_size)]
        test_labels = labels[train_size+valid_size:]

        assert data.iloc[-1]['label_id'].item() == test_labels[-1].item(), "The data is not split correctly."


        # In[22]:


        traindataset = TensorDataset(train_inputids, train_attention_masks, train_labels)
        validdataset = TensorDataset(valid_inputids, valid_attention_masks, valid_labels)
        testdataset = TensorDataset(test_inputids, test_attention_masks, test_labels)

        train_dataset = traindataset
        valid_dataset = validdataset
        test_dataset = testdataset

        # train_dataset, _ = random_split(traindataset, [train_size, 0])
        # valid_dataset, _ = random_split(validdataset, [valid_size, 0])
        # test_dataset, _ = random_split(testdataset, [test_size, 0])

        # DONT'T SHUFFLE DATA
        # train_dataset = Subset(traindataset, np.arange(train_size))
        # valid_dataset = Subset(validdataset, np.arange(valid_size))
        # test_dataset = Subset(testdataset, np.arange(test_size))

        print('train size: ', len(train_dataset))
        print('valid size: ', len(valid_dataset))
        print('test size: ', len(test_dataset))
        if not all([test_dataset[i][2] == test_labels[i] for i in range(test_size)]): 
            print("[PANIC] The data is shuffled.")


        # In[23]:


        type(train_dataset), type(traindataset)


        # In[31]:


        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

        batch_size = 64
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )
        validation_dataloader = DataLoader(
                    valid_dataset, # The validation samples.
                    sampler = SequentialSampler(valid_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )
        test_dataloader = DataLoader(
                    test_dataset, # The test samples.
                    sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )


        # In[32]:


        # Function to calculate the accuracy of our predictions vs labels
        def metric(preds, labels):
            _, predict_top1_id = torch.topk(preds, k=1, dim=1)
            predict_top1_id = predict_top1_id.numpy().flatten()
            assert len(predict_top1_id) == len(labels)
            top_1_acc = np.sum(predict_top1_id == labels) / len(labels)
            return top_1_acc # , top_5_acc

        def format_time(elapsed):
            '''
            Takes a time in seconds and returns a string hh:mm:ss
            '''
            # Round to the nearest second.
            elapsed_rounded = int(round((elapsed)))
            
            # Format as hh:mm:ss
            return str(datetime.timedelta(seconds=elapsed_rounded))


        # In[33]:


        import random
        import logging

        def set_seed(seed_val = 42):
            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            torch.cuda.manual_seed_all(seed_val)

        set_seed()
        log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

        def evaluate(dataloader, subset):
            logging.info("***** Running Evaluation *****")
            logging.info("Subset: {} Num steps = {}".format(subset, len(dataloader)))

            model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            all_logits = []
            all_label_ids = []
            all_predicted_classes = []
            for batch in dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                
                with torch.no_grad():
                    res = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                    loss = res['loss']
                    logits = res['logits']
                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                all_logits.append(logits.detach().cpu())
                all_label_ids.append(b_labels.to('cpu').numpy())
                all_predicted_classes.append(torch.argmax(logits, dim=1).to('cpu').numpy())
                # total_eval_accuracy += flat_accuracy(logits, label_ids)

            all_logits = torch.concat(all_logits, dim=0)
            all_label_ids = np.concatenate(all_label_ids, axis=0)
            all_predicted_classes = np.concatenate(all_predicted_classes, axis=0)
            # print(all_label_ids, len(all_label_ids), all_logits.shape)
            top1_acc = metric(all_logits, all_label_ids)
            logging.info("Subset: {} Top-1 Acc: {:.3f}".format(subset, top1_acc))
            avg_val_loss = total_eval_loss / len(dataloader)

            logging.info("Subset: {} Loss: {:.5f}".format(subset, avg_val_loss))
            logging.info("***** Evaluation Completed *****")
            
            return top1_acc, all_predicted_classes


        # In[34]:


        import numpy as np
        from tqdm import tqdm, trange
        # from torch.utils.tensorboard import SummaryWriter
        from transformers import get_linear_schedule_with_warmup
        from torch.optim import AdamW
        import os
        # set environment variable
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        ############## Model ################
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels = len(category_map), # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)


        ################ Arguments Starts #############
        epochs = 4
        save_steps = 500
        save_checkpoint = False
        learning_rate =  3e-5

        ################ Arguments Ends #############

        optimizer = AdamW(model.parameters(),
                        lr =learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )

        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        training_stats = []
        global_step = 0

        logging.info("Num train examples = %d", len(train_dataset))
        logging.info("Num Epochs = %d Total steps = %d", epochs, total_steps)
        model.zero_grad()

        evaluate(validation_dataloader, "Validation")
        evaluate(test_dataloader, "Testing")

        for epoch_i in range(0, epochs):
            total_train_loss = 0
            logging.info("***** Running training *****")
            model.train()
            logging.info("Epoch {} starts".format(epoch_i+1))
            for step, batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    logging.info('  Batch {}/{}, loss: {:.5f}'.format(step, len(train_dataloader), loss.item()))
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()        
                res = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
                loss = res['loss']
                logits = res['logits']
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                if save_checkpoint and global_step % save_steps == 0:
                    output_dir = os.path.join("./checkpoints", 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

            logging.info("Epoch {} completed".format(epoch_i+1))
            avg_train_loss = total_train_loss / len(train_dataloader)            
            logging.info("Average training loss: {:.2f}".format(avg_train_loss))
            
            # add support for roc/auc and f1 score
            top1_acc_train, _ = evaluate(train_dataloader, "Train")
            top1_acc_val, _ = evaluate(validation_dataloader, "Validation")
            top1_acc_test, _ = evaluate(test_dataloader, "Testing")
            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'top1_acc_train': top1_acc_train,
                    'top1_acc_val': top1_acc_val,
                    'top1_acc_test': top1_acc_test
                }
            )


        # In[ ]:


        # plot the train, validation and test accuracy w.r.t. epochs
        plt.clf()
        plt.plot([x['epoch'] for x in training_stats], [x['top1_acc_train'] for x in training_stats], label='Training')
        plt.plot([x['epoch'] for x in training_stats], [x['top1_acc_val'] for x in training_stats], label='Validation')
        plt.plot([x['epoch'] for x in training_stats], [x['top1_acc_test'] for x in training_stats], label='Testing')
        # do label
        plt.xlabel('Epoch')
        plt.ylabel('Top-1 Accuracy')
        plt.legend()
        graph_path = '../../../data/moral/graphs/'
        if not os.path.exists(graph_path):
            os.makedirs(graph_path)
        plt.savefig(graph_path + task_type+'_'+model_name.split("/")[-1].replace('-', '_')+'.png')

        largest_acc_epoch = np.argmax([x['top1_acc_val'] for x in training_stats])
        largest_top1_acc_val = training_stats[largest_acc_epoch]['top1_acc_val']
        largest_top1_acc_test = training_stats[largest_acc_epoch]['top1_acc_test']

        print("Largest top1_acc_val: {:.3f} at epoch {}".format(largest_top1_acc_val, largest_acc_epoch+1))

        top1_acc_test_final, preds_final_ids = evaluate(test_dataloader, "Final Testing")
        # map back to original labels
        category_map_rev = {v: k for k, v in category_map.items()}
        preds_final = [category_map_rev[pred] for pred in preds_final_ids]
        # get the categories back to test_data
        test_data['preds'] = preds_final

        # make sure the output directory exists
        output_path = '../../../data/moral/outputs/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        test_data.to_json(output_path + 'test_preds_'+task_type+'_'+model_name.split("/")[-1].replace('-', '_')+'.jsonl', orient='records', lines=True)
        
        model_path = '../../../data/moral/models/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save_pretrained(output_path + task_type+'_' + model_name.split("/")[-1].replace('-', '_'))
