
from abc import abstractmethod, ABCMeta

class Cache(metaclass=ABCMeta):

    def __init__(self, cache_dict=dict(), lock=None):
        self._dict = cache_dict
        self._idx = -1
        if lock is not None:
            self.Lock = lock

    @abstractmethod
    def initialize(self) -> bool:
        pass

    @abstractmethod
    def update(self, event):
        pass

    def getDict(self):
        return self._dict

    def isEmpty(self) -> bool:
        return not self._dict

    def __iter__(self):
        self._idx = -1
        return iter(self._dict)

    def __next__(self):
        self._idx += 1
        if self._idx < len(self._dict):
            return self._dict[self._idx]
        else:
            raise StopIteration

    def __getitem__(self, key):
        if key in self._dict:
            return self._dict[key]
        else:
            errStr = "Key [%s] not found in %s@[%s]." % (key, self.__class__, hex(id(self)))
            raise KeyError(errStr)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return "%s@[%s]:\n%s" % (self.__class__, hex(id(self)),[e for e in self._dict.values()])

    __repr__ = __str__