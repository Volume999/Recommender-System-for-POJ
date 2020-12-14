class FileSource:
    def __init__(self, file_path):
        self.file_path = file_path


class DataSourceCsv(FileSource):
    def __init__(self, file_path):
        FileSource.__init__(self, file_path)


class DataSourcePickle(FileSource):
    def __init__(self, file_path):
        FileSource.__init__(self, file_path)


class EnginePickle(FileSource):
    def __init__(self, file_path):
        FileSource.__init__(self, file_path)
