N=int(input()) #N명의 사람
total_time=0 #최소 토탈시간

#걸리는 시간 입력
time=list(map(int,input().split()))

#짧은 시간 순서대로 정렬
time=sorted(time) #1,2,3,3,4
#time.sort()
for i in range(1,N+1):
	total_time+=sum(time[0:i])

print(total_time)