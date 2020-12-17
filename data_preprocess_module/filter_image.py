import os

fileNumToDelete = []
for filename in os.listdir('labels/train2014'):
    if filename.endswith(".txt") or filename.endswith(".py"): 

        thisPath = os.path.join('labels/train2014', filename)
        img_num = filename.split('_')[2]
        img_num = img_num.split('.')[0]
        # print(img_num)

        fileToRead = open(thisPath, "r")
        lines = fileToRead.readlines()
        fileToRead.close()

        fileToEdit = open(thisPath, "w")
        for line in lines:
            if int(line.strip()[0]) == 0:
                fileToEdit.write(line)
        
        
        fileToEdit.close()
        
        fileToCheck = open(thisPath, "r")
        # print(len(fileToCheck.readlines()), img_num)
        if len(fileToCheck.readlines()) == 0:
            fileNumToDelete.append(img_num)
            
# print(fileNumToDelete)

for item in fileNumToDelete:
    txtFileName = "COCO_train2014_" + str(item) + ".txt"
    txtFilePath = os.path.join('labels/train2014', txtFileName)

    imgFileName = "COCO_train2014_" + str(item) + ".jpg"
    imgFilePath = os.path.join('images/train2014', imgFileName)

    os.remove(txtFilePath)
    os.remove(imgFilePath)

