n_li = []

for i in range(9):
	n = int(input())
	n_li.append(n)

num = max(n_li)

print(num)
print(int(n_li.index(num))+1)