import heapq as hq
import itertools

REMOVED = '<removed-item>'      # placeholder for a removed item

class PQueue:
    def __init__(self) -> None:
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of items to entries
        self.counter = itertools.count()     # unique sequence count

    def add_item(self, item, priority=0):
        'Add a new item or update the priority of an existing item'
        if item in self.entry_finder:
            self.remove_item(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        hq.heappush(self.pq, entry)

    def remove_item(self, item):
        'Mark an existing item as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(item)
        entry[-1] = REMOVED

    def popitem(self):
        'Remove and return the lowest priority item. Raise KeyError if empty.'
        while self.pq:
            priority, count, item = hq.heappop(self.pq)
            if item is not REMOVED:
                del self.entry_finder[item]
                return (item, priority)
        raise KeyError('pop from an empty priority queue')
    
    def __setitem__(self, key, item):
        self.add_item(key, item)

    def __getitem__(self, key):
        return self.entry_finder[key][0]
    
    def __contains__(self, key):
        return key in self.entry_finder

    def items(self):
        return self.entry_finder.items()
    
    def has_key(self, k):
        print(k)
        return k in self.entry_finder
