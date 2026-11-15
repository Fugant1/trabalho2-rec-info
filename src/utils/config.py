class Config:
    def __init__(self, config_dict):
        self.config = config_dict

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

    def to_dict(self):
        return self.config