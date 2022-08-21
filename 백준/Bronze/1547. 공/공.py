change = int(input())

# 첫번째 위치에 있는 숫자의 값을 출력
cups=[0,1,2,3]

for _ in range(change):
	x,y = map(int,input().split())
	cups[x],cups[y] = cups[y],cups[x]

print(cups.index(1))