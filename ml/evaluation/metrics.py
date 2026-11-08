import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix

def runs(labels):
    labels=list(labels)
    if not labels:return []
    starts=[0]+[i for i in range(1,len(labels)) if labels[i]!=labels[i-1]]
    return [(labels[s],s,e) for s,e in zip(starts,starts[1:]+[len(labels)])]
def edit_score(a,b):
    a=[x[0] for x in runs(a)];b=[x[0] for x in runs(b)];prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return 100*(1-prev[-1]/max(len(a),len(b),1))
def segment_f1(truth,pred,threshold):
    gt=runs(truth);ps=runs(pred);matched=set();tp=0
    for c,s,e in ps:
        options=[(max(0,min(e,b)-max(s,a))/(max(e,b)-min(s,a)),i) for i,(k,a,b) in enumerate(gt) if k==c and i not in matched]
        if options:
            score,i=max(options)
            if score>=threshold:matched.add(i);tp+=1
    return 2*tp/max(len(gt)+len(ps),1)
def evaluate(truth,pred,sample_fps=1.,latencies=None):
    if len(truth)!=len(pred) or not len(truth):raise ValueError("Aligned nonempty sequences required")
    p,r,f,_=precision_recall_fscore_support(truth,pred,labels=list(range(7)),zero_division=0)
    gt=runs(truth);ps=runs(pred);errors=[];used=set()
    # Greedy one-to-one, class-pair aware boundaries. Missed boundaries counted separately.
    for i in range(1,len(ps)):
        pair=(ps[i-1][0],ps[i][0]);options=[(abs(ps[i][1]-gt[j][1])/sample_fps,j) for j in range(1,len(gt)) if j not in used and (gt[j-1][0],gt[j][0])==pair]
        if options:err,j=min(options);errors.append(err);used.add(j)
    result={"accuracy":accuracy_score(truth,pred),"macro_f1":f1_score(truth,pred,labels=list(range(7)),average="macro",zero_division=0),"weighted_f1":f1_score(truth,pred,average="weighted",zero_division=0),"precision":p.tolist(),"recall":r.tolist(),"per_class_f1":f.tolist(),"confusion_matrix":confusion_matrix(truth,pred,labels=list(range(7))).tolist(),"segment_f1":{str(t):segment_f1(truth,pred,t) for t in (.1,.25,.5)},"edit_score":edit_score(truth,pred),"boundary_mae_seconds":float(np.mean(errors)) if errors else None,"missed_boundaries":len(gt)-1-len(used),"unmatched_predicted_boundaries":len(ps)-1-len(errors),"switches_per_minute":max(len(ps)-1,0)/(len(pred)/sample_fps/60),"segment_count_error":len(ps)-len(gt)}
    if latencies:result["latency_ms"]={"mean":float(np.mean(latencies)),"p95":float(np.percentile(latencies,95))};result["inference_fps"]=1000/np.mean(latencies)
    return result
