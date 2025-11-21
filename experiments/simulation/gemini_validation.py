import numpy as np
import pandas as pd

NUM_TRIALS = 2000
DOMAINS = ['Math', 'History', 'PopCulture']
TOLERANCE = 5.0

EXPERTISE = {
    'Agent_A': {'Math': 0.95, 'History': 0.10, 'PopCulture': 0.10},
    'Agent_B': {'Math': 0.60, 'History': 0.60, 'PopCulture': 0.60},
    'Agent_C': {'Math': 0.20, 'History': 0.20, 'PopCulture': 0.20},
}

CONFIDENCE_BIAS = {'Agent_A': 0.05, 'Agent_B': 0.05, 'Agent_C': 0.60}

class Agent:
    def __init__(self, name):
        self.name = name
        self.expertise = EXPERTISE[name]
        self.bias = CONFIDENCE_BIAS[name]
    
    def query(self, domain, correct_answer):
        true_skill = self.expertise[domain]
        noise = np.random.normal(0, (1.0 - true_skill) * 25)
        prediction = correct_answer + noise
        internal_uncertainty = np.clip((1.0 - true_skill) + 
np.random.normal(0, 0.05), 0.01, 0.99)
        reported_humility = np.clip(internal_uncertainty - self.bias, 
0.01, 0.99)
        return prediction, reported_humility

def majority_vote(predictions, humilities):
    return np.mean(predictions), 1 - np.mean([1-h for h in humilities])

def humility_weighted(predictions, humilities, domain, agents):
    h_adj = [humilities[i] * (1 - agents[i].expertise[domain]) for i in 
range(len(agents))]
    scores = np.array([(1-h)**2 for h in h_adj])
    weights = scores / np.sum(scores)
    pred = np.sum(np.array(predictions) * weights)
    base_H = np.sum(np.array(h_adj) * weights)
    final_H = np.clip(base_H + 0.5 * np.clip(np.std(predictions)/20, 0, 
1), 0.01, 0.99)
    return pred, final_H

def calc_opi(records):
    if not records: return 0, 0, 10
    df = pd.DataFrame(records)
    acc = df['correct'].mean()
    if acc < 0.01: return 0, 0, 10
    df['bin'] = pd.cut(df['confidence'], bins=10, labels=False)
    ece = sum(abs(df[df['bin']==i]['correct'].mean() - 
df[df['bin']==i]['confidence'].mean()) * len(df[df['bin']==i])/len(df) for 
i in range(10) if len(df[df['bin']==i])>0)
    return acc, ece, ece/acc

agents = [Agent('Agent_A'), Agent('Agent_B'), Agent('Agent_C')]
history = {k:[] for k in 
['Agent_A','Agent_B','Agent_C','Humility','Majority']}

print("Running 2000 trials...")
for _ in range(NUM_TRIALS):
    domain = np.random.choice(DOMAINS)
    preds, hums = [], []
    for agent in agents:
        p, h = agent.query(domain, 100.0)
        preds.append(p)
        hums.append(h)
        history[agent.name].append({'correct': abs(p-100)<TOLERANCE, 
'confidence': 1-h})
    
    ph, hh = humility_weighted(preds, hums, domain, agents)
    history['Humility'].append({'correct': abs(ph-100)<TOLERANCE, 
'confidence': 1-hh})
    
    pm, hm = majority_vote(preds, hums)
    history['Majority'].append({'correct': abs(pm-100)<TOLERANCE, 
'confidence': 1-hm})

print("\nRESULTS\n" + "="*60)
for name in ['Agent_A', 'Humility', 'Majority', 'Agent_B', 'Agent_C']:
    acc, ece, opi = calc_opi(history[name])
    print(f"{name:15s}  Acc:{acc:.3f}  ECE:{ece:.3f}  OPI:{opi:.3f}")

h_opi = calc_opi(history['Humility'])[2]
m_opi = calc_opi(history['Majority'])[2]
print(f"\nOPI Improvement: {((m_opi-h_opi)/m_opi*100):.1f}%")
print("✓ Complete")
