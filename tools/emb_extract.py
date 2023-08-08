# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import torch
import json
import numpy
import sys
import ipdb

def get_step_emb(input_step_list, output_emb_file, model_name="ViT-L/14"):
    # use CLIP langauge encoder to encode each sentence in order and save embeddings in to a file
    import clip
    prompts = [
        'a photo of {}.',
        'a photo of a person {}.',
        'a photo of a person using {}.',
        'a photo of a person doing {}.',
        'a photo of a person during {}.',
        'a photo of a person performing {}.',
        'a photo of a person practicing {}.',
        'a video of {}.',
        'a video of a person {}.',
        'a video of a person using {}.',
        'a video of a person doing {}.',
        'a video of a person during {}.',
        'a video of a person performing {}.',
        'a video of a person practicing {}.',
        'a example of {}.',
        'a example of a person {}.',
        'a example of a person using {}.',
        'a example of a person doing {}.',
        'a example of a person during {}.',
        'a example of a person performing {}.',
        'a example of a person practicing {}.',
        'a demonstration of {}.',
        'a demonstration of a person {}.',
        'a demonstration of a person using {}.',
        'a demonstration of a person doing {}.',
        'a demonstration of a person during {}.',
        'a demonstration of a person performing {}.',
        'a demonstration of a person practicing {}.',
    ]

    model, preprocess = clip.load(model_name) # clip.load("ViT-L/14") # clip.load("ViT-B/32")
    model.cuda().eval()
    model.float()

    text_features = []
    for x in input_step_list:
        # fill step descriptions into prompts
        sents = [prompt.format(x) for prompt in prompts]

        # CLIP langauge encoder
        text_tokens = clip.tokenize(sents, truncate=True).cuda()
        with torch.no_grad():
            sents_embs = model.encode_text(text_tokens).float().cpu()
            text_features.append(sents_embs.mean(0, keepdim=True))

    text_features = torch.cat(text_features)
    torch.save(text_features, output_emb_file)
    return text_features

def get_step_emb_lang(input_step_list, output_emb_file):
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer('paraphrase-mpnet-base-v2')

    text_features = []
    for x in input_step_list:
        with torch.no_grad():
            sents_embs = st_model.encode(x)
            sents_embs = torch.from_numpy(sents_embs)
            text_features.append(sents_embs.view(1, -1))

    text_features = torch.cat(text_features)
    torch.save(text_features, output_emb_file)
    return text_features

if __name__ == "__main__":
    # 5. HT100M steps (verb phrases) with "ViT-B/16"
    input_file = 'data/ht100m_asr_verb_phrase.txt'
    output_emb_file = 'data/clip_14_step_emb_ht100m_vbphrase.pth'
    
    step_vps = []
    with open(input_file, 'r') as f:
        for line in f:
            vp = line.strip()
            vp = vp.split(',')[0]
            step_vps.append(vp)
    print("We got {} verb phrases!".format(len(step_vps)))
    print(step_vps[:20])
    step_feat = get_step_emb(step_vps, output_emb_file, model_name="ViT-B/16")
    print("We got step embeddings with shape {} and save it to {}!".format(step_feat.size(), output_emb_file))
    
    ipdb.set_trace() # sys.exit(0)

    # 4. wikiHow steps (verb phrases)
    input_file = 'data/step_headline_verb_phrase.txt'
    output_emb_file = 'data/clip_step_emb_verb_phrase.pth'
    
    step_vps = []
    with open(input_file, 'r') as f:
        for line in f:
            vp = line.strip()
            step_vps.append(vp)
    print("We got {} verb phrases!".format(len(step_vps)))
    print(step_vps[:20])
    step_feat = get_step_emb(step_vps, output_emb_file)
    print("We got step embeddings with shape {} and save it to {}!".format(step_feat.size(), output_emb_file))
    
    ipdb.set_trace()
    
    
    # 3. COIN steps with Language BERT
    input_text_file = "data/step_coin_text.txt"
    output_emb_file = 'data/mpnet_step_emb_coin.pth'
    
    step_str_list = []
    with open(input_text_file, 'r') as f:
        for line in f:
            step_str = line.strip()
            step_str_list.append(step_str)
    print(step_str_list)
    print("We got {} step descriptions!".format(len(step_str_list)))  # 778 steps
    step_feat = get_step_emb_lang(step_str_list, output_emb_file)
    print("We got step embeddings with shape {} and saved it to {}!".format(step_feat.size(), output_emb_file))
    
    ipdb.set_trace()
    
    
    
    # 2. COIN steps with "ViT-L/14"
    input_text_file = "data/step_coin_text.txt"
    output_emb_file = 'data/clip_14_step_emb_coin.pth'
    
    step_str_list = []
    with open(input_text_file, 'r') as f:
        for line in f:
            step_str = line.strip()
            step_str_list.append(step_str)
    print(step_str_list)
    print("We got {} step descriptions!".format(len(step_str_list)))  # 778 steps
    step_feat = get_step_emb(step_str_list, output_emb_file)
    print("We got step embeddings with shape {} and saved it to {}!".format(step_feat.size(), output_emb_file))
    
    ipdb.set_trace()
    
    
    
    # 1. wikiHow steps
    input_text_file = "data/step_label_text.json"
    input_text = json.load(open(input_text_file)) # a list
    output_text_file = "data/step_headline_text.txt"
    output_emb_file = 'data/clip_step_emb_only_headline.pth'

    # the mapping in paper [10588, 1059]; used to get task id
    # ref_map = torch.load('data/step_to_task.pth') 
    # row_col_inds = ref_map.nonzero()

    # get a list of step headlines and save them into a text file
    step_headlines = []
    step_to_task = []
    for task_i, task in enumerate(input_text):
        for step in task:
            headline = step['headline']
            headline = headline.replace('\n', '').strip() # remove special tokens
            # if headline == '':
            #     continue
            if len(headline) > 0:
                headline = headline[0].lower() + headline[1:] # lower the first char
            step_headlines.append(headline)
            step_to_task.append(task_i)

    with open(output_text_file, 'w') as f:
        for headline in step_headlines:
            f.write(headline + '\n')

    # generate the mapping from steps into tasks
    # map_mtx = torch.zeros(len(step_headlines), len(input_text)).long() # [10588, 1053]
    # map_mtx[torch.arange(len(step_headlines)), torch.tensor(step_to_task)] = 1
    print("We got {} tasks!".format(len(input_text)))  # 1053 articles
    print("We got {} headlines!".format(len(step_headlines)))  # 10588 steps
    step_feat = get_step_emb(step_headlines, output_emb_file)
    print("We got step embeddings with shape {} and save it to {}!".format(step_feat.size(), output_emb_file))






