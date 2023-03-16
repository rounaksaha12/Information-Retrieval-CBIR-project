import pickle

file = open("bean-features.pickle",'rb')
object_file = pickle.load(file)
file.close()

print(object_file)