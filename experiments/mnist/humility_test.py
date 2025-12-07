import random

print("HUMILITY PROTOCOL TEST")
print("="*50)

class Agent:
    def __init__(self, name, skill):
        self.name = name
        self.skill = skill
    
    def predict(self, answer):
        noise = random.gauss(0, 10 * (1 - self.skill))
        pred = answer + noise
        hum = 1 - self.skill
        return pred, hum

a = Agent("Good", 0.9)
b = Agent("OK", 0.6)
c = Agent("Bad", 0.1)

base_ok = 0
hum_ok = 0

for i in range(100):
    truth = random.uniform(0, 100)
    pa, ha = a.predict(truth)
    pb, hb = b.predict(truth)
    pc, hc = c.predict(truth)
    
    base = (pa + pb + pc) / 3
    
    wa = (1 - ha) ** 2
    wb = (1 - hb) ** 2
    wc = (1 - hc) ** 2
    total = wa + wb + wc
    hum = (pa * wa + pb * wb + pc * wc) / total
    
    if abs(base - truth) < 2:
        base_ok += 1
    if abs(hum - truth) < 2:
        hum_ok += 1

print("Baseline:", base_ok, "%")
print("Humility:", hum_ok, "%")
print("Improvement:", hum_ok - base_ok, "%")
