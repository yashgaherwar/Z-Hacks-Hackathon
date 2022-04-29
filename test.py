

f = open("face_result.txt", "r")
array = []
for i in f.readlines():
    array.append(i[:-1])

print(max(array, key = array.count))
