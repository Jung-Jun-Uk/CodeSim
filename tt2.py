import os

path = 'code/'
dir = os.listdir(path)

print(len(dir))

for d in dir:            
    path2 = os.path.join(path,d)
    scripts = os.listdir(path2)
    # print(path2, len(scripts))
    
    # print(len(scripts))    