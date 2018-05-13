import os
import json
from nltk import sent_tokenize, word_tokenize


def tokenize(st):
    #TODO: The tokenizer's performance is suboptimal
    ans = []
    for sent in sent_tokenize(st):
        ans += word_tokenize(sent)
    nst = " ".join(ans).lower()
    ans.clear()
    for sent in sent_tokenize(nst):
        ans += word_tokenize(sent)

    return ans

if __name__ == "__main__":
    difficulty_set = ["middle", "high"]
    data = "data"
    raw_data = "data/RACE"
    cnt = 0
    avg_article_length = 0
    avg_question_length = 0
    avg_option_length = 0
    num_que = 0
    for data_set in ["test", "dev", "train"]:#, "dev", "train"

        p1 = os.path.join(data, data_set)
        if not os.path.exists(p1):
            os.mkdir(p1)
        for d in difficulty_set:
            samples = []
            new_data_path = os.path.join(data, data_set, d)
            if not os.path.exists(new_data_path):
                os.mkdir(new_data_path)
            new_raw_data_path = os.path.join(raw_data, data_set, d)
            for inf in os.listdir(new_raw_data_path):
                cnt += 1
                sample = {}
                #print(cnt)
                obj = json.load(open(os.path.join(new_raw_data_path, inf), "r"))
                obj['article'] = obj['article'].replace("\\newline", "\n")
                obj['article'] = tokenize(obj['article'])
                sample['article'] = obj['article']
                avg_article_length += obj['article'].count(" ")
                for i in range(len(obj['questions'])):
                    num_que += 1
                    obj['questions'][i] = tokenize(obj['questions'][i])
                    sample['questions'] = obj['questions'][i]
                    sample['answer'] = obj['answers'][i]
                    avg_question_length += obj['questions'][i].count(" ")
                    for k in range(4):
                        obj['options'][i][k] = tokenize(obj['options'][i][k])
                        sample['option'+str(k)] = obj['options'][i][k]
                        avg_option_length += obj['options'][i][k].count(" ")
                #json.dump(obj, open(os.path.join(new_data_path, inf), "w"), indent=4)
                    samples.append(sample.copy())
                    #print(sample)

            json.dump(samples,  open(os.path.join(data, str(data_set + '_' + d +".json")), "w"))

    '''print "avg article length", avg_article_length * 1. / cnt
    print "avg question length", avg_question_length * 1. / num_que
    print "avg option length", avg_option_length * 1. / (num_que * 4)'''
