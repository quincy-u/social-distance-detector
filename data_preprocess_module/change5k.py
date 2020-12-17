import os 

file5k = open('5k.txt', 'r')
file5kLine = file5k.readlines()
fileTrainVal = open('trainvalno5k.txt', 'r')
fileTrainValLine = fileTrainVal.readlines()
file5k.close()
fileTrainVal.close()


f2 = []
for filename in os.listdir('images/train2014'):
    if filename.endswith(".jpg") or filename.endswith(".py"): 
    	f2.append(str(filename + '\n'))

f4 = []
for filename in os.listdir('images/val2014'):
    if filename.endswith(".jpg") or filename.endswith(".py"): 
    	f4.append(str(filename + '\n'))


print(file5kLine[4].split('/')[-1] in f4)
print(len(f2))
print(len(f4))

fiveK = open('5k.txt', 'w')
fiveKTrainVal = open('trainvalno5k.txt', 'w')

a = 0
b = 0
for line in file5kLine:
	num = line.split('/')[-1]
	# print(num)
	if num in f4:
		a+=1
		fiveK.write(line)
fiveK.close()

for line in fileTrainValLine:
	num = line.split('/')[-1]
	if num in f2:
		b+=1
		fiveKTrainVal.write(line)
fiveKTrainVal.close()

print(a)
print(b)