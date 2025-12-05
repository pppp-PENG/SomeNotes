# Java Basic Grammar

### Arrays

```java
Arrays.copyOfRange(T[], int from, int to)	// return T[]
Arrays.sort(T[], int from, int to)			// no return
```

### String

### ArrayList

### LinkedList

```java
LinkedList<E> ll = new LinkedList<>();
ll.size();
ll.addFirst(E);			// no return
ll.addLast(E);			// no return
ll.getFirst(E);			// return E
ll.getLast(E);			// return E
ll.removeFirst(E);		// return E
ll.removeLast(E);		// return E
ll.toArray(T[] a);		// return T[]
```

### HashSet<E>

```java
Set<E> set = new HashSet<>();
set.size();			// return int
set.add(E);			// return boolean
set.remove(E);		// return boolean
set.contains(E);	// return boolean
set.isEmpty();		// return boolean
set.toArray(T[] a);	// return T[]
```

### HashMap<K, V>

```java
Map<K, V> map = new HashMap<>();
map.size();				// return int;
map.put(K, V);			// return V
map.remove(K);			// return V
map.get(K);				// return K
map.isEmpty();			// return boolean
map.containsKey(K);		// return boolean
map.containsValue(V);	// return boolean
map.keySet();			// return Set<K>
map.values();			// return Collection<V>
```

