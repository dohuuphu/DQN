b = {'a':{}, 'b':{}}
list_key = ['a', 'b']
a = dict(dict.fromkeys(list_key, dict()))
print(type(a), type(b))
a['a'].update({'c':1})
b['a'].update({'c':1})
print(a)