from yaml import load, SafeLoader
from os import path, environ
from re import compile
from .cache import Cache

class ConfigDict(Cache):

    __slots__='_dict','_curSection'

    def __init__(self, cache_dict, curSection=''):
        super().__init__(cache_dict=cache_dict)
        self._curSection = curSection

    def initialize(self) -> bool:
        pass

    def update(self, event):
        pass

    def contains(self, key : str) -> bool:
        return str(key) in self._dict

    def _get(self, key : str):
        try:
            return self[key]
        except KeyError:
            errStr = "Key [%s] not found in config %s." % (key, '' if self._curSection == '' else "  {'%s': %s}" % (self._curSection, self._dict))
            raise KeyError(errStr)

    def get_value(self, key, default_value=None):
        try:
            value = self._get(key)
            return str(value)
        except KeyError as err:
            if default_value is None:
                raise ValueError("%s  No default_value provided" % err)
            return default_value
            
    def expand(self, key):
        try:
            value = self._get(key)
            return ConfigDict(value, key)
        except KeyError:
            return ConfigDict(dict(), key)
      
    def __str__(self):
        return "{%s: %s}" % (self._curSection, str(self._dict))

class YamlLoader(SafeLoader):
    path_matcher = compile(r'(\$\{([^}^{]+)\})')

    def __init__(self, stream):

        self._root = path.split(stream.name)[0]
        
        super(YamlLoader, self).__init__(stream)

        YamlLoader.add_constructor('!include', YamlLoader.include)
        YamlLoader.add_constructor('!path', YamlLoader.path_constructor)
        YamlLoader.add_implicit_resolver('!path', YamlLoader.path_matcher, None)

    @classmethod
    def load(cls, config) -> ConfigDict:
        with open(config) as yaml:
            return ConfigDict(load(yaml, YamlLoader))

    def include(self, node) -> ConfigDict:
        filename = YamlLoader.path_constructor(self, node)
        with open(filename, 'r') as f:
            return load(f, YamlLoader)

    def path_constructor(loader, node):
        ''' Extract the matched value, expand env variable, and replace the match '''
        value = node.value
        match = YamlLoader.path_matcher.match(value)
        env_var = match.group()[2:-1]
        return environ.get(env_var) + value[match.end():]