# coding=utf-8
import requests
import json
import time
import re
import datetime
import pandas as pd
from bs4 import BeautifulSoup
from snownlp import SnowNLP
import collections
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import numpy as np

#文本摘要
def split_sentence(text, punctuation_list='!?。！？'):
    """
    将文本段安装标点符号列表里的符号切分成句子，将所有句子保存在列表里。
    """
    sentence_set = []
    inx_position = 0         #索引标点符号的位置
    char_position = 0        #移动字符指针位置
    for char in text:
        char_position += 1
        if char in punctuation_list:
            next_char = list(text[inx_position:char_position+1]).pop()
            if next_char not in punctuation_list:
                sentence_set.append(text[inx_position:char_position])
                inx_position = char_position
    if inx_position < len(text):
        sentence_set.append(text[inx_position:])

    sentence_with_index = {i:sent for i,sent in enumerate(sentence_set)} #dict(zip(sentence_set, range(len(sentences))))
    return sentence_set,sentence_with_index

def get_tfidf_matrix(sentence_set,stop_word):
    corpus = []
    for sent in sentence_set:
        sent_cut = jieba.cut(sent)
        sent_list = [word for word in sent_cut if word not in stop_word]
        sent_str = ' '.join(sent_list)
        corpus.append(sent_str)

    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    # word=vectorizer.get_feature_names()
    tfidf_matrix=tfidf.toarray()
    return np.array(tfidf_matrix)

def get_sentence_with_words_weight(tfidf_matrix):
    sentence_with_words_weight = {}
    for i in range(len(tfidf_matrix)):
        sentence_with_words_weight[i] = np.sum(tfidf_matrix[i])

    max_weight = max(sentence_with_words_weight.values()) #归一化
    min_weight = min(sentence_with_words_weight.values())
    for key in sentence_with_words_weight.keys():
        x = sentence_with_words_weight[key]
        sentence_with_words_weight[key] = (x-min_weight)/(max_weight-min_weight)

    return sentence_with_words_weight

def get_sentence_with_position_weight(sentence_set):
    sentence_with_position_weight = {}
    total_sent = len(sentence_set)
    for i in range(total_sent):
        sentence_with_position_weight[i] = (total_sent - i) / total_sent
    return sentence_with_position_weight

def similarity(sent1,sent2):
    """
    计算余弦相似度
    """
    return np.sum(sent1 * sent2) / 1e-6+(np.sqrt(np.sum(sent1 * sent1)) *\
                                    np.sqrt(np.sum(sent2 * sent2)))

def get_similarity_weight(tfidf_matrix):
    sentence_score = collections.defaultdict(lambda :0.)
    for i in range(len(tfidf_matrix)):
        score_i = 0.
        for j in range(len(tfidf_matrix)):
            score_i += similarity(tfidf_matrix[i],tfidf_matrix[j])
        sentence_score[i] = score_i

    max_score = max(sentence_score.values()) #归一化
    min_score = min(sentence_score.values())
    for key in sentence_score.keys():
        x = sentence_score[key]
        sentence_score[key] = (x-min_score)/(max_score-min_score)

    return sentence_score

def ranking_base_on_weigth(sentence_with_words_weight,
                            sentence_with_position_weight,
                            sentence_score, feature_weight = [1,1,1]):
    sentence_weight = collections.defaultdict(lambda :0.)
    for sent in sentence_score.keys():
        sentence_weight[sent] = feature_weight[0]*sentence_with_words_weight[sent] +\
                                feature_weight[1]*sentence_with_position_weight[sent] +\
                                feature_weight[2]*sentence_score[sent]

    sort_sent_weight = sorted(sentence_weight.items(),key=lambda d: d[1], reverse=True)
    return sort_sent_weight

def get_summarization(sentence_with_index,sort_sent_weight,topK_ratio =0.3):
    topK = int(len(sort_sent_weight)*topK_ratio)
    summarization_sent = sorted([sent[0] for sent in sort_sent_weight[:topK]])

    summarization = []
    for i in summarization_sent:
        summarization.append(sentence_with_index[i])

    summary = ''.join(summarization)
    return summary

#引入停用词
path = 'cn_stopwords.txt'
f = open(path, 'r', encoding='UTF-8',errors = 'ignore')
stop = f.read()
stop_word = re.split(r'[\n]', stop)

#逐句情感计算
def cal_emo(test):
    poemstr = ''
    poem_sentence_list = []
    for i in test:
        poemstr += i
    poem_sentence_list = re.split(r'[，。！？]', poemstr)
    del(poem_sentence_list[-1])
    print(poem_sentence_list)
    emotion = 0
    lens = 0
    for sentence in poem_sentence_list:
        if len(sentence) > 1:
            s = SnowNLP(sentence)
            emotion += s.sentiments
            lens += 1
    emotion = emotion / lens
    return emotion

def get_data(url):
    '''
    功能：访问 url 的网页，获取网页内容并返回
    参数：
        url ：目标网页的 url
    返回：目标网页的 html 内容
    '''
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    }

    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.text

    except requests.HTTPError as e:
        print(e)
        print("HTTPError")
    except requests.RequestException as e:
        print(e)
    except:
        print("Unknown Error !")

def parse_data(html):
    '''
    功能：提取 html 页面信息中的关键信息，并整合一个数组并返回
    参数：html 根据 url 获取到的网页内容
    返回：存储有 html 中提取出的关键信息的数组
    '''
    json_data = json.loads(html)['data']
    comments = [['name', 'gender','follower_count','voteup','cmt_count','content', 'character_num', 'emotion1', 'relief', 're1']]

    try:
        for item in json_data:
            comment = []
            comment.append(item['author']['name'])    # 姓名
            comment.append(item['author']['gender'])  # 性别
            comment.append(item['author']['follower_count'])     # 粉丝数
            comment.append(item['voteup_count'])      # 点赞数
            comment.append(item['comment_count'])     # 评论数
            raw = BeautifulSoup(item['content']).get_text()
            comment.append(raw)            # 回答
            comment.append(len(raw))
            #comment.append(item)                       # all
            s = SnowNLP(raw).sentiments
            comment.append(s)            # 回答
            #comment.append(cal_emo(raw))            # 回答
            if len(raw) > 400:
                sentence_set,sentence_with_index = split_sentence(raw, punctuation_list='!?。！？')
                tfidf_matrix = get_tfidf_matrix(sentence_set,stop_word)
                sentence_with_words_weight = get_sentence_with_words_weight(tfidf_matrix)
                sentence_with_position_weight = get_sentence_with_position_weight(sentence_set)
                sentence_score = get_similarity_weight(tfidf_matrix)
                sort_sent_weight = ranking_base_on_weigth(sentence_with_words_weight,
                                                            sentence_with_position_weight,
                                                            sentence_score, feature_weight = [1,1,1])
                r = 300 / len(raw)
                summarization = get_summarization(sentence_with_index,sort_sent_weight,topK_ratio = r)

            else:
                summarization = raw
            comment.append(summarization)            # 回答
            rs = SnowNLP(summarization).sentiments
            comment.append(rs)            # 回答
            #comment.append(cal_emo(summarization))            # 回答
            comments.append(comment)

        return comments

    except Exception as e:
        print(comment)
        print(e)

def save_data(comments, name):
    '''
    功能：将comments中的信息输出到文件中/或数据库中。
    参数：comments 将要保存的数据
    '''
    filename = name + '.csv'

    dataframe = pd.DataFrame(comments)
    dataframe.to_csv(filename, mode='a', index=False, sep=',', header=False)
    #dataframe.to_csv(filename, mode='a', index=False, sep=',', header=['name','gender','follower_count','voteup','cmt_count','content', 'character_num', 'emotion1', 'emotion2', 'relief', 're1', 're2'])


    '''
    with open(filename, "a", encoding='utf-8') as f:
        f.write(json.dumps(comments, ensure_ascii=False, indent = 4, separators=(',', ':')))
    '''

def merge(num, name):

    url = 'https://www.zhihu.com/api/v4/questions/' + str(num) +'/answers?include=data%5B%2A%5D.is_normal%2Cadmin_closed_comment%2Creward_info%2Cis_collapsed%2Cannotation_action%2Cannotation_detail%2Ccollapse_reason%2Cis_sticky%2Ccollapsed_by%2Csuggest_edit%2Ccomment_count%2Ccan_comment%2Ccontent%2Ceditable_content%2Cvoteup_count%2Creshipment_settings%2Ccomment_permission%2Ccreated_time%2Cupdated_time%2Creview_info%2Crelevant_info%2Cquestion%2Cexcerpt%2Crelationship.is_authorized%2Cis_author%2Cvoting%2Cis_thanked%2Cis_nothelp%2Cis_labeled%3Bdata%5B%2A%5D.mark_infos%5B%2A%5D.url%3Bdata%5B%2A%5D.author.follower_count%2Cbadge%5B%2A%5D.topics&limit=5&offset=5&platform=desktop&sort_by=default'

    # get total cmts number
    html = get_data(url)
    totals = json.loads(html)['paging']['totals']

    print(totals)
    print('---'*10)

    page = 0

    while(page < totals):
        url = 'https://www.zhihu.com/api/v4/questions/' + str(num) +'/answers?include=data%5B%2A%5D.is_normal%2Cadmin_closed_comment%2Creward_info%2Cis_collapsed%2Cannotation_action%2Cannotation_detail%2Ccollapse_reason%2Cis_sticky%2Ccollapsed_by%2Csuggest_edit%2Ccomment_count%2Ccan_comment%2Ccontent%2Ceditable_content%2Cvoteup_count%2Creshipment_settings%2Ccomment_permission%2Ccreated_time%2Cupdated_time%2Creview_info%2Crelevant_info%2Cquestion%2Cexcerpt%2Crelationship.is_authorized%2Cis_author%2Cvoting%2Cis_thanked%2Cis_nothelp%2Cis_labeled%3Bdata%5B%2A%5D.mark_infos%5B%2A%5D.url%3Bdata%5B%2A%5D.author.follower_count%2Cbadge%5B%2A%5D.topics&limit=5&offset='+ str(page) +'&platform=desktop&sort_by=default'

        html = get_data(url)
        comments = parse_data(html)
        save_data(comments, name)

        print(page)
        page += 5


if __name__ == '__main__':
    merge(50832612, '生物博士毕业后都从事什么工作')
    print("完成！！")
