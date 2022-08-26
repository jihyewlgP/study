score=[]

for _ in range(5):
	a,b,c,d=map(int,input().split())
	s=a+b+c+d
	score.append(s)

m_s=max(score)

print((score.index(m_s))+1, m_s)