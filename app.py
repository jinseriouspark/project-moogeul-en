import datetime 
import numpy as np
import pandas as pd
import re
import json
import os
import glob

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from transformers import BertModel
from transformers import AutoTokenizer

import argparse
from bs4 import BeautifulSoup
import requests

def split_essay_to_sentence(origin_essay):
    origin_essay_sentence = sum([[a.strip() for a in i.split('.')] for i in origin_essay.split('\n')], [])
    essay_sent = [a for a in origin_essay_sentence if len(a) > 0]
    return essay_sent

def get_first_extraction(text_sentence):
    row_dict = {}
    for row in tqdm(text_sentence):
        question = 'what is the feeling?'
        answer = question_answerer(question=question, context=row)
        row_dict[row] = answer
    return row_dict


class myDataset_for_infer(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        sentences =  tokenizer(self.X[idx], return_tensors = 'pt', padding = 'max_length', max_length = 96, truncation = True)
        return sentences
    
    
def infer_data(model, main_feeling_keyword):
    #ds = myDataset_for_infer()
    df_infer = myDataset_for_infer(main_feeling_keyword)

    infer_dataloader = torch.utils.data.DataLoader(df_infer, batch_size= 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        model = model.cuda()

    result_list = []
    with torch.no_grad():
        for idx, infer_input in tqdm(enumerate(infer_dataloader)):
            mask = infer_input['attention_mask'].to(device)
            input_id = infer_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            result = np.argmax(output.logits, axis=1).numpy()
            result_list.extend(result)
    return result_list


def get_word_emotion_pair(cls_model, origin_essay_sentence, idx2emo):

    import re
    def get_noun(sent):
        return [w for (w, p) in pos_tag(word_tokenize(p_texts[0])) if len(w) > 1 and p in (['NN','N','NP'])]
    def get_adj(sent):
        return [w for (w, p) in pos_tag(word_tokenize(p_texts[0])) if len(w) > 1 and p in (['ADJ'])]
    def get_verb(sent):
        return [w for (w, p) in pos_tag(word_tokenize(p_texts[0])) if len(w) > 1 and p in (['VERB'])]

    result_list = infer_data(cls_model, origin_essay_sentence)
    final_result = pd.DataFrame(data = {'text': origin_essay_sentence , 'label' : result_list})
    final_result['emotion'] = final_result['label'].map(idx2emo)
    
    final_result['noun_list'] = final_result['text'].map(get_noun)
    final_result['adj_list'] = final_result['text'].map(get_adj)
    final_result['verb_list'] = final_result['text'].map(get_verb)
    
    final_result['title'] = 'none'
    file_made_dt = datetime.datetime.now()
    file_made_dt_str = datetime.datetime.strftime(file_made_dt, '%Y%m%d_%H%M%d')
    os.makedirs(f'./result/{nickname}/{file_made_dt_str}/', exist_ok = True)
    final_result.to_csv(f"./result/{nickname}/{file_made_dt_str}/essay_result.csv", index = False)

    return final_result, file_made_dt_str

    return final_result, file_made_dt_str


def get_essay_base_analysis(file_made_dt_str, nickname):
    essay1 = pd.read_csv(f"./result/{nickname}/{file_made_dt_str}/essay_result.csv")
    essay1['noun_list_len'] = essay1['noun_list'].apply(lambda x : len(x))
    essay1['noun_list_uniqlen'] = essay1['noun_list'].apply(lambda x : len(set(x)))
    essay1['adj_list_len'] = essay1['adj_list'].apply(lambda x : len(x))
    essay1['adj_list_uniqlen'] = essay1['adj_list'].apply(lambda x : len(set(x)))
    essay1['vocab_all'] = essay1[['noun_list','adj_list']].apply(lambda x : sum((eval(x[0]),eval(x[1])), []), axis=1)
    essay1['vocab_cnt'] = essay1['vocab_all'].apply(lambda x : len(x))
    essay1['vocab_unique_cnt'] = essay1['vocab_all'].apply(lambda x : len(set(x)))
    essay1['noun_list'] = essay1['noun_list'].apply(lambda x : eval(x))
    essay1['adj_list'] = essay1['adj_list'].apply(lambda x : eval(x))
    d = essay1.groupby('title')[['noun_list','adj_list']].sum([]).reset_index()
    d['noun_cnt'] = d['noun_list'].apply(lambda x : len(set(x)))
    d['adj_cnt'] = d['adj_list'].apply(lambda x : len(set(x)))

    # 문장 기준 최고 감정
    essay_summary =essay1.groupby(['title'])['emotion'].value_counts().unstack(level =1)

    emo_vocab_dict = {}
    for k, v in essay1[['emotion','noun_list']].values:
      for vocab in v:
        if (k, 'noun', vocab) not in emo_vocab_dict:
          emo_vocab_dict[(k, 'noun', vocab)] = 0

        emo_vocab_dict[(k, 'noun', vocab)] += 1

    for k, v in essay1[['emotion','adj_list']].values:
      for vocab in v:
        if (k, 'adj', vocab) not in emo_vocab_dict:
          emo_vocab_dict[(k, 'adj', vocab)] = 0

        emo_vocab_dict[(k, 'adj', vocab)] += 1
    vocab_emo_cnt_dict = {}
    for k, v in essay1[['emotion','noun_list']].values:
      for vocab in v:
        if (vocab, 'noun') not in vocab_emo_cnt_dict:
          vocab_emo_cnt_dict[('noun', vocab)] = {}
        if k not in vocab_emo_cnt_dict[( 'noun', vocab)]:
          vocab_emo_cnt_dict[( 'noun', vocab)][k] = 0

        vocab_emo_cnt_dict[('noun', vocab)][k] += 1

    for k, v in essay1[['emotion','adj_list']].values:
      for vocab in v:
        if ('adj', vocab) not in vocab_emo_cnt_dict:
          vocab_emo_cnt_dict[( 'adj', vocab)] = {}
        if k not in vocab_emo_cnt_dict[( 'adj', vocab)]:
          vocab_emo_cnt_dict[( 'adj', vocab)][k] = 0

        vocab_emo_cnt_dict[('adj', vocab)][k] += 1

    vocab_emo_cnt_df = pd.DataFrame(vocab_emo_cnt_dict).T
    vocab_emo_cnt_df['total'] = vocab_emo_cnt_df.sum(axis=1)
    # 단어별 최고 감정 및 감정 개수
    all_result=vocab_emo_cnt_df.sort_values(by = 'total', ascending = False)

    # 단어별 최고 감정 및 감정 개수 , 형용사 포함 시
    adj_result=vocab_emo_cnt_df.sort_values(by = 'total', ascending = False)

    # 명사만 사용 시
    noun_result=vocab_emo_cnt_df[vocab_emo_cnt_df.index.get_level_values(0) == 'noun'].sort_values(by = 'total', ascending = False)

    final_file_name = f"essay_all_vocab_result.csv"
    adj_file_name = f"essay_adj_vocab_result.csv"
    noun_file_name = f"essay_noun_vocab_result.csv"
    
    os.makedirs(f'./result/{nickname}/{file_made_dt_str}/', exist_ok = True)
    
    all_result.to_csv(f"./result/{nickname}/{file_made_dt_str}/essay_all_vocab_result.csv", index = False)
    adj_result.to_csv(f"./result/{nickname}/{file_made_dt_str}/essay_adj_vocab_result.csv", index = False)
    noun_result.to_csv(f"./result/{nickname}/{file_made_dt_str}/essay_noun_vocab_result.csv", index = False)
    
    return all_result, adj_result, noun_result, essay_summary, file_made_dt_str



from transformers import AutoModelForSequenceClassification
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def all_process(origin_essay, nickname):
    essay_sent =split_essay_to_sentence(origin_essay)
    idx2emo = {0: 'Anger', 1: 'Sadness', 2: 'Anxiety', 3: 'Hurt', 4: 'Embarrassment', 5: 'Joy'}
    tokenizer = AutoTokenizer.from_pretrained('seriouspark/xlm-roberta-base-finetuning-sentimental-6label')
    cls_model = AutoModelForSequenceClassification.from_pretrained('seriouspark/xlm-roberta-base-finetuning-sentimental-6label')
    
    final_result, file_name_dt = get_word_emotion_pair(cls_model, essay_sent, idx2emo)
    all_result, adj_result, noun_result, essay_summary, file_made_dt_str = get_essay_base_analysis(file_name_dt, nickname)
    
    summary_result = pd.concat([adj_result, noun_result]).fillna(0).sort_values(by = 'total', ascending = False).fillna(0).reset_index()[:30]
    with open(f'./result/{nickname}/{file_name_dt}/summary.json','w') as f:
        json.dump( essay_summary.to_json(),f)
    with open(f'./result/{nickname}/{file_made_dt_str}/all_result.json','w') as f:
        json.dump( all_result.to_json(),f)    
    with open(f'./result/{nickname}/{file_made_dt_str}/adj_result.json','w') as f:
        json.dump( adj_result.to_json(),f)  
    with open(f'./result/{nickname}/{file_made_dt_str}/noun_result.json','w') as f:
        json.dump( noun_result.to_json(),f)  
    #return essay_summary, summary_result
    total_cnt = essay_summary.sum(axis=1).values[0]
    essay_summary_list = sorted(essay_summary.T.to_dict()['none'].items(), key = lambda x: x[1], reverse =True)
    essay_summary_list_str = ' '.join([f'{row[0]} {int(row[1]*100 / total_cnt)}%' for row in essay_summary_list])
    summary1 = f"""{nickname}, Your sentiments in your writting are [{essay_summary_list_str}] """

    return summary1

def get_similar_vocab(message):
    if (len(message) > 0) & (len(re.findall('[A-Za-z]+', message))> 0):
        vocab = message
        all_dict_url = f"https://www.dictionary.com/browse/{vocab}"
        response = requests.get(all_dict_url)

        html_content = response.text
        # BeautifulSoup로 HTML 파싱
        soup = BeautifulSoup(html_content, 'html.parser')
        result = soup.find_all(class_='ESah86zaufmd2_YPdZtq')
        p_texts = [p.get_text() for p in soup.find_all('p')]
        whole_vocab = sum([ [word for word , pos in pos_tag(word_tokenize(text)) if pos in ['NN','JJ','NNP','NNS']] for text in p_texts],[])

        similar_words_final = Counter(whole_vocab).most_common(10)
        return [i[0] for i in similar_words_final]
    
    else: 
        return message
    
def get_similar_means(vocab):
    all_dict_url = f"https://www.dictionary.com/browse/{vocab}"
    response = requests.get(all_dict_url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    result = soup.find_all(class_='ESah86zaufmd2_YPdZtq')
    p_texts = [p.get_text() for p in soup.find_all('p')]
    return p_texts[:10]


info_dict = {}
def run_all(message, history):
    global info_dict
    if message.find('NICKNAME:')>=0:
        global nickname 
        nickname = message.replace('NICKNAME','').replace(':','').strip()
        #global nickname
        info_dict[nickname] = {}
        return f'''Good [{nickname}]!! Let's start!. 
Give me a vocabulary in your mind.
\n\n\nwhen you type the vocab, please include \"VOCAB: \" 
e.g <VOCAB: orange>
'''
    try :
        #print(nickname)
        if message.find('VOCAB:')>=0:
            clear_message = message.replace('VOCAB','').replace(':','').strip()
            info_dict[nickname]['main_word'] = clear_message
            vocab_mean_list = []
            similar_words_final = get_similar_vocab(clear_message)
            print(similar_words_final)
            similar_words_final_with_main = similar_words_final + [clear_message]
            if len(similar_words_final_with_main)>0:
                for w in similar_words_final_with_main:
                    temp_means = get_similar_means(w)
                    vocab_mean_list.append(temp_means[:2])
                fixed_similar_words_final = list(set([i for i in sum(vocab_mean_list, []) if len(i) > 10]))[:10]


                word_str = ' \n'.join([str(idx) + ") " + i for idx, i in enumerate(similar_words_final, 1)])
                sentence_str = ' \n'.join([str(idx) + ") " + i for idx, i in enumerate(fixed_similar_words_final, 1)])

                return f'''Let's start writing with the VOCAB<{clear_message}>! 
First, how about those similar words?
{word_str} \n
The word has these meanings. 
{sentence_str}\n
Pick and type one meaning of these list.
\n\n\n When you type in, please include \"SENT:\", like this.
\n e.g. <SENT: a globose, reddish-yellow, bitter or sweet, edible citrus fruit. >
    '''
            else:
                return 'Include \"VOCAB:\" please (VOCAB: orange)'

        elif message.find('SENT:')>=0:
            clear_message = message.replace('SENT','').replace(':','').strip()
            info_dict[nickname]['selected_sentence'] = clear_message
            return f'''You've got [{clear_message}]. 
\n With this sentence, we can make creative short writings
\n\n\n Include \"SHORT_W: \", please.
\n e.g <SHORT_W: Whenever I smell the citrus, I always reminise him, first>

            '''

        elif message.find('SHORT_W:')>=0:
            clear_message = message.replace('SHORT_W','').replace(':','').strip()
            info_dict[nickname]['short_contents'] = clear_message

            return f'''This is your short sentence <{clear_message}> .
\n With this sentence, let's step one more thing, please write long sentences more than 500 words.
\n\n\n When you input, please include\"LONG_W: \" like this.
\n e.g <LONG_W: He enjoyed wearing blue T-shirts at the gym, but the intense citrus scent he used on his clothes was noticeably excessive  ... >
            '''
        elif message.find('LONG_W:')>=0:
            long_message = message.replace('LONG_W','').replace(':','').strip()

            length_of_lm = len(long_message)
            if length_of_lm >= 500:
                info_dict['long_contents'] = long_message
                os.makedirs(f"./result/{nickname}/", exist_ok = True)
                with open(f"./result/{nickname}/contents.txt",'w') as f:
                    f.write(long_message)
                return f'Your entered text is {length_of_lm} characters. This text is worth analyzing. If you wish to start the analysis, please type "START ANALYSIS"'
            else :
                return f'The text you have entered is {length_of_lm} characters. It\'s a bit short for analysis. Could you please provide a bit more sentences'

        elif message.find('START ANALYSIS')>=0:
            with open(f"./result/{nickname}/contents.txt",'r') as f:
                    orign_essay = f.read()
            summary = all_process(orign_essay, nickname)
            
            #print(summary)
            return summary
        else:
            return 'Please start from the beginning'

    except:
        return 'An error has occurred. Restarting from the beginning. Please enter your NICKNAME:'       
                
        
import gradio as gr
import requests
history = []
info_dict = {}
iface = gr.ChatInterface(
    fn=run_all,
    chatbot = gr.Chatbot(),
    textbox = gr.Textbox(placeholder="Please enter including the chatbot's request prefix.", container = True, scale = 7),
    title = 'MooGeulMooGeul',
    description = "Please start by choosing your nickname. Include 'NICKNAME: ' in your response",
    theme = 'soft',
    examples = ['NICKNAME: bluebottle',
                'VOCAB: orange',
                'SENT: a globose, reddish-yellow, bitter or sweet, edible citrus fruit.',
                'SHORT_W: Whenever I smell the citrus, I always reminise him, first',
                '''LONG_W: Whenever I smell citrus, I always think of him. He used to come to the gym wearing a blue T-shirt, often spraying a strong citrus scent.
                That scent was quite distinctive, letting me know when he was passing by. 
                I usually arrived to work out between 7:00 and 7:30 AM, and interestingly, he would arrive about 10 minutes after me.
                On days I came early, he did too; and when I was late, he was also late.
                The citrus scent from his body was always so intense, as if he had just sprayed it.'''
               ],
    cache_examples = False,
    retry_btn = None,
    undo_btn = 'Delete Previous',
    clear_btn = 'Clear',
                   
)
iface.launch(share=True)