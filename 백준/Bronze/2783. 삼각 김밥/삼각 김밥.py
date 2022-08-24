money=[]

Y,X = map(float,input().split())

one_m = Y/X
money.append(one_m)

N = int(input())

for i in range(N):
	Y,X = map(float,input().split())
	one_m = Y/X
	money.append(one_m)

l = min(money)*1000
print(round(l,2))