from file_handler import read_text


class Dataloader:
    def get_data(self):
        pass


class FileDataloader(Dataloader):
    def __init__(self, file_name):
        self.file_name = file_name

    def get_data(self):
        lines = read_text(self.file_name)
        return self.process_lines(lines)

    def process_lines(self, lines):
        return lines


class SentenceDataloader(FileDataloader):
    def __init__(self, file_name):
        super().__init__(file_name)

    def process_lines(self, lines):
        # import regex as re
        data = []
        for line in lines:
            # line = re.sub(r'(\p{P})', r' \1 ', line.lower())
            # line = re.sub(r'\s{2,}', ' ', line)
            data.append(line)
        return data
