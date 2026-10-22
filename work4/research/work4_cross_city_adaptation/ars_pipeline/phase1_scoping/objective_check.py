# -*- coding: utf-8 -*-
"""Exact-arithmetic risk-objective examples; no dataset access."""
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path

checks = []
def check(name, condition):
    checks.append({'name': name, 'passed': bool(condition)})
    if not condition:
        raise AssertionError(name)

def mean(xs):
    return sum(xs) / len(xs)

risks = {'reference': [Q('.1'), Q('.4')], 'A': [Q('.08'), Q('.35')], 'B': [Q('.12'), Q('.30')]}
baseline = risks['reference']
shifted = {k: [v-b for v,b in zip(row,baseline)] for k,row in risks.items()}
oracle = [min(row[i] for row in risks.values()) for i in range(2)]
regret = {k: [v-b for v,b in zip(row,oracle)] for k,row in risks.items()}
winner = lambda data, reducer: min(data,key=lambda k: reducer(data[k]))
check('Mean baseline subtraction is a parameter-independent constant', all(mean(shifted[k]) == mean(row)-mean(baseline) for k,row in risks.items()))
check('Mean risk and mean increment have same minimizer', winner(risks,mean) == winner(shifted,mean) == 'B')
check('Worst raw and baseline-shifted risk may differ', winner(risks,max) == 'B' and winner(shifted,max) == 'A')
check('Fixed-reference increment differs from oracle regret', winner(regret,max) == 'B' and winner(shifted,max) == 'A')
check('Oracle regret nonnegative while reference increment may be negative', all(v>=0 for row in regret.values() for v in row) and max(shifted['A'])<0)
check('Simplex maximum equals largest shifted group loss', all(max(Q(i,100)*row[0]+(1-Q(i,100))*row[1] for i in range(101))==max(row) for row in shifted.values()))
check('Feasible reference gives zero observed-group increment', max(shifted['reference'])==0 and min(map(max,shifted.values()))<=0)
target_risks = {'reference':Q('.1'),'A':Q('.3'),'B':Q('.05')}
check('Observed improvements do not guarantee arbitrary target improvement', max(shifted['A'])<0 and target_risks['A']>target_risks['reference'])

def cross(xs,ps,ys,qs):
    return sum(p*q*abs(x-y) for x,p in zip(xs,ps) for y,q in zip(ys,qs))

def crps(xs,ps,y):
    return sum(p*abs(x-y) for x,p in zip(xs,ps))-cross(xs,ps,xs,ps)/2

x0,p0=[Q(0),Q(1)],[Q(1,2),Q(1,2)]
x1,p1=[Q(1,4),Q(3,4)],[Q(1,3),Q(2,3)]
D=cross(x0,p0,x1,p1)-(cross(x0,p0,x0,p0)+cross(x1,p1,x1,p1))/2
check('Energy half-distance nonnegative',D>=0)
for y in [Q(0),Q(1,5),Q(1),Q(3,2)]:
    for lam in [Q(0),Q(1,4),Q(1,2),Q(3,4),Q(1)]:
        direct=crps(x0+x1,[(1-lam)*p for p in p0]+[lam*p for p in p1],y)
        formula=(1-lam)*crps(x0,p0,y)+lam*crps(x1,p1,y)-lam*(1-lam)*D
        check('Exact mixture CRPS y=%s lambda=%s'%(y,lam),direct==formula)
out={'status':'PASS','check_count':len(checks),'checks':checks,
     'scope':'finite rational examples and algebra only; not traffic experiment or generalization proof',
     'data_access':'no traffic data; no target labels; no GPU',
     'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'risks':{k:list(map(str,v)) for k,v in risks.items()},
     'selected':{'mean_risk':winner(risks,mean),'worst_risk':winner(risks,max),'worst_reference_increment':winner(shifted,max),'worst_oracle_regret':winner(regret,max)},
     'mixture_identity':'L(F_lambda,y)=(1-lambda)L(F0,y)+lambda L(F1,y)-lambda(1-lambda)D',
     'new_method_established':False}
Path(__file__).with_name('OBJECTIVE_CHECK_RESULTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','checks':len(checks),'selected':out['selected']}))
