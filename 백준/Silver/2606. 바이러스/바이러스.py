n=int(input())#컴퓨터수
l=int(input())#연결수

graph=[[]*(n+1) for _ in range(n+1)]

for _ in range(l):
	#노드연결
	n1,n2 = map(int,input().split())
	graph[n1].append(n2)
	graph[n2].append(n1)

visited=[0]*(n+1) # 방문 노드 표시

#dfs
def dfs(start):
    global count
    visited[start] = 1
	
    for i in graph[start]:
        if visited[i]==0:
            dfs(i)

dfs(1)
print(visited.count(1)-1) # 시작점을 제외 #바이러스에 감염된 수