import sys

allLabels = ['A','B','C','D','E','F','G','H','I','J']
nbLines = 2000
fd =open(sys.argv[1],'r')
lines = fd.readlines()

count=0
for label in lines:
	if label.strip() in allLabels:
		count+=1
	else:
		if count<nbLines:
			print("Wrong label line:"+str(count+1))
			break
if count<nbLines:
	print("Labels Check : fail!",count,nbLines)
else:
	print("Labels Check : Successfull!")


