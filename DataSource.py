class FileSource:
    def __init__(self, file_path):
        self.file_path = file_path


# Database exported in CSV
class DataSourceCsv(FileSource):
    def __init__(self, file_path):
        FileSource.__init__(self, file_path)


# Saving the engine in bytes for faster read next time
class EnginePickle(FileSource):
    def __init__(self, file_path):
        FileSource.__init__(self, file_path)
