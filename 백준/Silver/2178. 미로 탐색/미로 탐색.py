import sys
from collections import deque
input = sys.stdin.readline

n, m = tuple(map(int, input().split()))
graph = [list(map(int, input().strip())) for _ in range(n)]
visited = [[0 for _ in range(m)] for _ in range(n)]

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
def bfs(x, y):
    q = deque([(x, y, 1)])
    visited[x][y] = True
    while q:
        x, y, distance = q.popleft()
        if x == n-1 and y == m-1:
            print(distance)
            break
        else:
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny] and graph[nx][ny] == 1:
                    visited[nx][ny] = True
                    q.append((nx, ny, distance + 1))
bfs(0, 0)