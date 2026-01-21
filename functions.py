import jieba
import re
import jieba.posseg as pseg
import os
import emoji

def add_replacements(rules_file, replacement_rules):
    with open(rules_file, 'r', encoding='utf-8') as f:
        for line in f:
            word, replacement = line.strip().split('，')
            replacement_rules[word.strip()] = replacement.strip()
    return replacement_rules

def add_words_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            jieba.add_word(word)


def replace_w_rules(tokens,replacement_rules):
    for i in range(len(tokens)):
        if tokens[i] in replacement_rules:
            tokens[i] = replacement_rules[tokens[i]]   
        else:
            continue

def connect_adj_noun(tokens):
    connected_tokens = []
    i = 0
   
    while i < len(tokens):
        word, flag = tokens[i]
        adj = flag == 'a'
        try:
            noun = tokens[i + 1][1] in ['n','vn', 'nr', 'nt', 'nz']
        except:
            print(tokens)
            print(i)
            pass
        more = i + 1 < len(tokens)
        single = len(word)==1
        if adj and noun and more and single:
            connected_word = word + tokens[i + 1][0]
            connected_tokens.append((connected_word, 'an'))  # Using 'an' for adjective-noun
            i += 1  # Skip the next token as it's already connected
        else:
            connected_tokens.append((word, flag))
        i += 1
        
        if i+1>len(tokens):
            break
        elif i+1==len(tokens):
            word, flag = tokens[i]
            connected_tokens.append((word, flag))
            break
            
    return connected_tokens

def replace_pronouns(tokens):
    recent_noun = None
    possessive = ""
    replaced_tokens = []
    
    for word, flag in tokens:
        if flag in ['n', 'nr', 'nt', 'nz']:
            if possessive:
                recent_noun = possessive + word
                possessive = ""
            else:
                recent_noun = word
            if word in ['父母','幼崽','孩子','妈妈','母亲']:
                replaced_tokens.append((recent_noun, flag))
            else:
                replaced_tokens.append((word, flag))
        elif word in ['我', '你', '他们','她们']:
            possessive = word + '的'
            replaced_tokens.append((word, flag))
        elif word in ['他', '她', '它', 'ta']:
            if recent_noun:
                replaced_tokens.append((recent_noun, 'r'))  # Replace pronoun with the recent noun
            else:
                replaced_tokens.append((word, flag))  # If no recent noun, keep pronoun
        else:
            replaced_tokens.append((word, flag))
    
    return replaced_tokens


def preprocess_chinese_text(text, replacement_rules, mode=''):
    # remove emojis
    text = emoji.demojize(text)
    emoji_mapping = {
    r":cow_face::horse_face:": "牛马",
    "doge": "狗头表情"
    }
    for eng_desc, zh_desc in emoji_mapping.items():
        text = re.sub(eng_desc, zh_desc, text)
    
    # delete 回复 @ XXXX：
    text = re.sub(r'回复|[\u4e00-\u9fff]@[^:]*:','', text)
    
    # remove @XXXX
    text = re.sub('@[^:]* ','', text) 
    if len(text)<10: text = re.sub('@[^:]*','', text)
    
    # remove number bullets
    text = re.sub('[\d].', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenize text using jieba
    jieba.setLogLevel(20)
    tokens = [(word.word, word.flag) for word in pseg.cut(text)]
    
    # connect single character adjective with following noun
    # tokens = connect_adj_noun(tokens)
    
    # replace some pronouns with its referents
    # tokens = replace_pronouns(tokens)
    
    # keep nouns, verbs, adjectives, i, x, and maybe others
    tokens = [word for word, flag in tokens if flag in ['n', 'v', 'a','i', 'x', 'c', 't','r','an','vn','nr']]
    
    # remove spaces
    tokens = [token for token in tokens if token!=' ']
    
    # Remove stopwords
    stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8')] 
    # tokens = [token for token in tokens if token not in stopwords]
    
    # Remove adverbs & quantities
    filtered_words = [line.strip() for line in open('filtered_words.txt', 'r', encoding='utf-8')] 
    tokens = [token for token in tokens if token not in filtered_words]
    
    # Remove low frequency words
    low_frequency = [line.strip() for line in open('low_frequency.txt', 'r', encoding='utf-8')] 
    tokens = [token for token in tokens if token not in low_frequency]
    
    # substitute synonyms
    replace_w_rules(tokens, replacement_rules)
    
    if mode=='pca':
        return " ".join(tokens)
    elif mode =='':
        return tokens
