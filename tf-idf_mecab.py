import subprocess
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer


def mecab(text):
    P = subprocess.Popen(["mecab","-d","/usr/local/lib/mecab/dic/mecab-ipadic-neologd"],
        stdin = subprocess.PIPE,
        stdout =subprocess.PIPE
    )
    P.stdin.write(text)
    P.stdin.close()
    result = P.stdout.read()
    result = result.decode("utf-8")
    return result

result =[]
noun=[]
train_texts=[]
for line in open("test.txt","rb"):
    r =mecab(line)
    r=r.strip()
    result.append(r)
#print(result)

for detail in result:
    detail=detail.replace("\t",",")
    detail=detail.replace(","," ")
    Res= detail.split("\n")
    noun.append(Res)
#print(noun)
#noun=list(itertools.chain.from_iterable(noun))
#print(noun)

for item in noun:
    #print(item)
    text=[]
    for i in item:
        #print(i)
        i=i.split(" ",10)
        #print(i)
        if i[0]=="EOS":
            break
        if i[1]=="名詞":
        #if i[1] == "名詞" and len(i[0]) != 1 and i[2] != "代名詞" and i[2] != "非自立" and i[2] != "接尾" and i[2] != "数" and i[2] != "特殊" and i[3] != "人名" and i[3] != "地域" and i[2] != "副詞可能":
            text.append(i[0])
            #print(i)
        """
        if i[1]=="動詞":
            text.append(i[0])
        """
        """
        if i[1]=="形容詞":
            text.append(i[0])
        """
    train_texts.append(text)
#print(train_texts)
#print(text)

docs=[]
tx=""
for i in train_texts:
    #print(i)
    tx=""
    for x in i:
        tx +=x
        tx +=" "
    docs.append(tx)
#------------------------------------
#print(docs)

vectorizer = TfidfVectorizer(max_df=0.9) # tf-idfの計算
#                            ^ 文書全体の90%以上で出現する単語は無視する
X = vectorizer.fit_transform(docs)
#print('feature_names:', vectorizer.get_feature_names())

words = vectorizer.get_feature_names()
for doc_id, vec in zip(range(len(docs)), X.toarray()):
    #print('doc_id:', doc_id)
    print("----------------------------")
    for w_id, tfidf in sorted(enumerate(vec), key=lambda x: x[1], reverse=True):
        #print(w_id)
        lemma = words[w_id]

        if tfidf==0:
            break

        #print('\t{0:s}, {1:f}'.format(lemma, tfidf))
        print('{0:s}, {1:f}'.format(lemma, tfidf))
        #f.write(str(lemma)+" : "+str(tfidf)+"\n")
