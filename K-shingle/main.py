import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np

from tqdm import tqdm
import gc
import os

def k_shingles(k,text,lib):
    """
        Convert each review into a set of k-shingles
        Input:
            k: number of shingles
            text: one review text
            lib: shingle dict
        Return:
            tset: dense representation of k-shingle for one review
    """
    b=0
    tset=set()
    while (b+k)<=len(text):
        str=text[b:b+k]
        p=37**(k-1)
        sum=0
        for s in str:
            sum+=lib[s]*p
            p/=37
        tset.add(int(sum))
        b+=1
    return np.array(list(tset))

def clean_data(vtext,stop_w,cd):
    """
    remove stopwords and punctuation marks
    Input:
        vtext: Original text
        stop_w: Stop words should removed
        cd: Characters and numbers should keeped
    Retruen:
        ntext: New text without stop words and punctuation marks, in list
    """
    text=[]
    for t in vtext:
        tnew=""
        res=[]
        for w in t.lower():
            if w in cd:
                tnew+=w
            if w==' ':
                if tnew in stop_w:
                    tnew=""
                else:
                    res.append(tnew)
                    tnew=""
        if not tnew in stop_w:
            res.append(tnew)
        text.append(res)
    ntext=[]
    for line in text:
        ntext.append(" ".join(line))
    return ntext

def jaccard(a,b):
    """
    Calculate jaccard distance
    Input:s
        a: set a
        b: set b
    Output:
        jac: jaccard distance
    """
    a=set(a.tolist())
    b=set(b.tolist())
    return 1-float(len(a&b))/float(len(a|b))

def minhash(a,b,ab,bb,p,pb,set):
    """
    Calculate a band hash value for each band
    Input:
        a:hash para for signature
        ab:hash para for hash band
        b:hash para for signature
        bb:hash para for hash band
        p:hash mod factor
        pb: hash mod factor for hash band
        set: data set
    Return:
        r:band hash value
    """
    
    r=(a*set+b)%p#hash for single signature
    r=r.min()
    return (r*ab+bb)%pb#hash for the band

def Search(stop_w,cd,shingle_lib,thn,hpa_a,hpa_b,hb_a,hb_b,p,pb,hbn,hnpb,banddic,context_mtx,nrev,c_id):
    """
    Search for a closest review ID for a text
    Input:
        stop_w:stop words
        cd:saved words
        shingle_lib:library for k shingle
        thn:total hash function number
        hpa_a: random set 1
        hpa_b: random set 2
        hb_a: random set 3
        hb_b: random set 4
        p:large prime
        pb: large prime 2
        hbn: hash band number
        hnpb: hash number per band
        banddic: list for saving similar review indexes, each element is a dic, whose key is hash band value, value is set of all similar review indexes
        context_mtx: kshingled text
        nrev: new review
        c_id: id list
    Return:
        closeset review ID
    """
    nrev=clean_data([nrev],stop_w,cd)
    nrev=nrev[0]
    nrev=k_shingles(4,nrev,shingle_lib)
    if len(nrev)==0:
        print("The most similar review ID is: None")
        return
    bandnv=[]    
    sum=0
    bind=0
    for i in range(thn):
        sum+=minhash(hpa_a[i],hpa_b[i],hb_a[bind],hb_b[bind],p,pb,nrev)
        bind+=1
        if i%hnpb==(hnpb-1):#10 hash function each band
            bandnv.append(sum)
            sum=0
            bind=0
    simrev=set()
    for i in range(len(bandnv)):
        if bandnv[i] in banddic[i].keys():
            simrev=simrev|banddic[i][bandnv[i]]
    jd=float("inf")
    res=None
    for i in simrev:
        dis=jaccard(nrev,context_mtx[i])
        if dis<0.2 and dis<jd:
            res=c_id[i]
            jd=dis
    print("The most similar review ID is:",res)

def main():

    ###Problem 1:
    #read data as list
    print("Problem 1")
    all = pd.read_json("amazonReviews.json", encoding='utf-8', lines=True)
    c_id=all['reviewerID'].values.tolist()
    c_text=all['reviewText'].values.tolist()
    #stop words and characters & numbers
    stop_w={"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}
    cd={'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9'}
    #remove stopwords and punctuation marks
    c_text=clean_data(c_text,stop_w,cd)

    ###Problem 2:
    print("Problem 2")
    shingle_lib={'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,'u':20,'v':21,'w':22,'x':23,'y':24,'z':25,'0':26,'1':27,'2':28,'3':29,'4':30,'5':31,'6':32,'7':33,'8':34,'9':35,' ':36}
    context_mtx=[]
    for text in c_text:
        context_mtx.append(k_shingles(4,text,shingle_lib))
    
    del all,text# empty the memory (too many variables)
    gc.collect()

    ###Problem 3:
    print("Problem 3")
    np.random.seed(0)
    sum=0
    y=[]
    min=float("inf")
    for i in range(10000):
        r1=np.random.randint(0,len(context_mtx)-1,size=1)
        r2=np.random.randint(0,len(context_mtx)-1,size=1)
        j=jaccard(context_mtx[r1[0]],context_mtx[r2[0]])
        sum+=j
        if j<min:
            min=j
        y.append(j)
    ave=sum/10000
    print("min j=",min,"\n","ave j=",ave)
    plt.hist(y, edgecolor='k')
    plt.xlabel('Jaccard distance')  
    plt.ylabel('Numbers of pair')  
    plt.title('Histogram of evaluated pairs') 
    plt.show()

    ###Problem 4
    print("Problem 4")
    hbn=30     #band numbers
    hnpb=10     #hash numbers per band
    thn=hbn*hnpb #total hash numbers
    p=2147482949       #Prime
    pb=2000001  #prime for band
    banddic=[]    #new data structure, has 30 bands and 15w data. Each band, which is a dictionary, contains the buckets which is a hash band value and contains many IDs
    for i in range(hbn):
        banddic.append({})

    np.random.seed(10)
    hpa_a=np.random.randint(1,p-1,size=thn,dtype=np.int64) # a in signature hash function
    np.random.seed(11)
    hpa_b=np.random.randint(0,p-1,size=thn,dtype=np.int64)# b in signature hash function
    np.random.seed(1)
    hb_a=np.random.randint(0,pb-1,size=hnpb,dtype=np.int64)# a in band hash function
    np.random.seed(7)
    hb_b=np.random.randint(0,pb-1,size=hnpb,dtype=np.int64)# b in signature hash function

    
    for txs in tqdm(range(len(context_mtx))):
        if len(context_mtx[txs])==0:
            continue
        sum=0
        bind=0
        for i in range(thn):
            sum+=minhash(hpa_a[i],hpa_b[i],hb_a[bind],hb_b[bind],p,pb,context_mtx[txs])
            bind+=1
            if i%hnpb==(hnpb-1):#10 hash function each band
                if not sum in banddic[i//10]: # I directly put the indexes in bucket, and all buckets in a band (dictionary) is a element of banddic list. For saving time and space
                    banddic[i//10][sum]={txs}# save the index instead of strings
                else:
                    banddic[i//10][sum].add(txs)
                sum=0
                bind=0
   

    ###Problem 5
    print("Problem 5")
    resdic={}

    for i in tqdm(banddic):
        for j in i.keys():
            if len(i[j])<2:
                continue
            for m in i[j]:
                if not m in resdic:
                    resdic[m]=set()
                for n in i[j]:
                    if n!=m and jaccard(context_mtx[m],context_mtx[n])<0.2:
                        resdic[m].add(n)
                if len(resdic[m])==0:
                    del resdic[m]

    v = list(resdic.values())
    kn=[]
    vn=[]
    for i in range(len(v)):
        if len(v[i])>1:
            vv=list(v[i])
            for ii in range(len(vv)):
                for jj in range(ii,len(vv)):
                    if vv[ii]!=vv[jj]:
                        vn.append([c_id[vv[ii]],c_id[vv[jj]]])
                        kn.append([c_text[vv[ii]],c_text[vv[jj]]])

    df = pd.DataFrame(list(zip(vn,kn)),columns=['ID', 'Similar_set'])
    df.to_csv('result.csv')
    del vn,kn,v,c_text# empty the memory (too many variables)
    gc.collect()
    ###Problem 6
    print("Problem 6")
    while True:
        print("Input your review (input 'end' to end function):")
        nrev=input()
        if nrev=="end":
            break
        Search(stop_w,cd,shingle_lib,thn,hpa_a,hpa_b,hb_a,hb_b,p,pb,hbn,hnpb,banddic,context_mtx,nrev,c_id)


if __name__ == '__main__':
    main()
    