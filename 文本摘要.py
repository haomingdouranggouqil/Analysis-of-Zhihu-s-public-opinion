  # coding:utf-8
import jieba
import re
import numpy as np
import collections
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


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


if __name__ == '__main__':
    #引入停用词
    path = 'cn_stopwords.txt'
    f = open(path, 'r', encoding='UTF-8',errors = 'ignore')
    stop = f.read()
    stop_word = re.split(r'[\n]', stop)
    text = '校内有食堂和超市，也能正常收发快递，外卖也能到宿舍区门口拿，可以说基本的生存条件还是具备的。但是据我所见，除本校学生外，对教职员工和部分外籍学生几乎没有管控措施。在校门口观察到，这些人群几乎随意出校，进校时顶多测量一下体温，也不需要随申码和行动轨迹，车辆也几乎是随意进出。而如果你是本校学生，你出一次校门需要自己打印申请表，写明出行原因和返校时间，到某办公室找负责老师签字和盖 章，然后出校前出示该申请表，才允许放行，再次进校后需要上交申请表。真是感谢学校，据说还组织了校外的理发师进校，在规定时间内可以到校内某处剪头发。另外在如此防控措施之下，竟然经常能在校内看到 明显的“非校内人员”，有大妈带着孩子来玩的，甚至还有外校的团体来参观或是进行活动。'
    sentence_set,sentence_with_index = split_sentence(text, punctuation_list='!?。！？')
    tfidf_matrix = get_tfidf_matrix(sentence_set,stop_word)
    sentence_with_words_weight = get_sentence_with_words_weight(tfidf_matrix)
    sentence_with_position_weight = get_sentence_with_position_weight(sentence_set)
    sentence_score = get_similarity_weight(tfidf_matrix)
    sort_sent_weight = ranking_base_on_weigth(sentence_with_words_weight,
                                                sentence_with_position_weight,
                                                sentence_score, feature_weight = [1,1,1])

    r = 300 / len(text)
    summarization = get_summarization(sentence_with_index,sort_sent_weight,topK_ratio = r)
    print('summarization:\n',summarization)
