m=[]
count=0
n,k=map(int,input().split())

for i in range(n):
	m.append(int(input()))

for i in reversed(range(n)):
	if k==0:
		break

	if k >= m[i]:
		count+=k//m[i]
		k=k%m[i]

print(count)