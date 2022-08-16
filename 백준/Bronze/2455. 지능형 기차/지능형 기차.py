t=0
max_t=[]

while(1):
	s,b = map(int,input().split())
	
	if b == 0:
		break
	else:
		t += b-s
		max_t.append(t)
		
print(max(max_t))