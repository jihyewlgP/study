N = int(input())

graph=[list(map(int,input())) for _ in range(N)]
num=[]

cnt = 0
# 상하좌우
dx = [-1,1,0,0]
dy = [0,0,-1,1]

#dfs 깊이 우선 탐색
def dfs(x,y):
    global cnt
    if x<0 or x>=N or y<0 or y>=N:
        return False
    
    if graph[x][y]==1:
        cnt +=1
        graph[x][y] = 0 #방문 위치 0 변환
        
        for i in range(4):
            dfs(x+dx[i],y+dy[i])
        return True

for i in range(N):
    for j in range(N):
        if dfs(i,j) == True:
            num.append(cnt) # 동 수 추가
            cnt = 0
            
print(len(num)) # 아파트 수
num.sort()
for i in num:
    print(i)