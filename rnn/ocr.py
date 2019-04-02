import gzip
import csv
import numpy as np

class OcrDataset:

    URL = 'http://ai.stanford.edu/~btaskar/ocr/letter.data.gz'
    
    def __init__(self, cache_dir):
        return super().__init__(*args, **kwargs) 


    def _read(filepath):
        with gzip.open(filepath, 'rt') as file_:
            reader = csv.reader(file_, delimiter = '\t')
            lines = list(reader)
            return lines
    
    def _parse(lines):
        lines = sorted(lines,key = lambda x :int(x[0]))
        data, target = [], []
        next_ = None
        for line in lines:
            if not next_:
                data.append([])
                target.append([])
            else:
                assert next_ = int(line[0])
            next_ = int(line[2]) if int(line[2] > -1 ) else None

            pixels = np.array([int(x) for x in line[6:134]])
            pixels = pixels.reshape((16,8))
            data[-1].append(pixels)
            target[-1].append(line[1])
        return data, target

    def _pad(data,target):
        max_length = max(len(x) for x in target)
        padding = np.zeros((16,8))
        data = [x + ([padding] * (max_length - len(x))) for x in data]
        target = [x + ([''] * (max_length - len(x))) for x in target]
        return np.array(data), np.array(target)