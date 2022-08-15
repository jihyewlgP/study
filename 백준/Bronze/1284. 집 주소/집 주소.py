while (1):
	num = input()
	if num=='0':
		break
	l= len(num)+1
	for i in num:
		if i == '0':
			l+=4
		elif i == '1':
			l+=2
		else:
			l+=3
	print(l)