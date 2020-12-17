import os

fileNumToDelete = []
count = 0
for filename in os.listdir('labels/val2014'):
    if count % 2 == 0: 
        

        thisPath = os.path.join('labels/val2014', filename)
        img_num = filename.split('_')[2]
        img_num = img_num.split('.')[0]
        # print(img_num)
            
        fileNumToDelete.append(img_num)
    count += 1

# print(fileNumToDelete)

for item in fileNumToDelete:
    txtFileName = "COCO_val2014_" + str(item) + ".txt"
    txtFilePath = os.path.join('labels/val2014', txtFileName)

    imgFileName = "COCO_val2014_" + str(item) + ".jpg"
    imgFilePath = os.path.join('images/val2014', imgFileName)

    os.remove(txtFilePath)
    os.remove(imgFilePath)
