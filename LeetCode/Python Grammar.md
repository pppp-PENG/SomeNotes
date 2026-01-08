```python
# sort & sorted
X.sort(key, reverse: bool)
sorted(iterable, key, reverse: bool)
'''
examples:
sorted([[1, 3], [2, 2]], key=lambda x: x[1]) ==> [[2, 2], [1, 3]]
'''

# execute the string
eval(expression: str) -> obj
'''
examples:
eval("6 * 2") ==> 12
eval("{'name': 'Tom', 'age': 20}") ==> {'name': 'Tom', 'age': 20}
'''

# accumulate operation
functools.reduce(function, iterable) -> obj
'''
function: the operation
iterable: the iterable object

examples:
reduce(add, [1, 2, 3, 4]) ==> 10
reduce(lambda x, y: x * y, [1, 2, 3, 4]) ==> 24
'''

# iter all the permutations
itertools.permutations(iterable, r=None) -> iter
'''
iterable: the iterable object
r: length of every permutation

examples:
perms = permutations([1, 2, 3])
for perm in perms:
	print(perm)
==> (1, 2, 3) (1, 3, 2) (2, 1, 3) (2, 3, 1) (3, 1, 2) (3, 2, 1)
'''
```

