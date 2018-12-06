class Ngram:
    def __init__(self, n):
        self.n = n

    def tmp_ngram(self, x):
        return (list(ngrams(x, self.n)) if len(x) >= self.n else [])


def get_ngrams(sentences, n, use_pool_thread=True):
    ng = Ngram(n)
    if use_pool_thread:
        local_ngramgs = Threader(sentences, ng.tmp_ngram).run()
    else:
        local_ngramgs = [ng.tmp_ngram(sentence) for sentence in sentences]
    return local_ngramgs


class Threader:
    def __init__(self, items, function, proc_num=None, show_tqdm=False):
        self.items = items
        self.function = function
        self.show_tqdm = show_tqdm
        if proc_num is None:
            proc_num = os.cpu_count()
        self.proc_num = proc_num
        self.pool = Pool(proc_num)
        self.total_size = len(items)
        self.batch_size = int(self.total_size / proc_num)
        if self.batch_size == 0:
            self.batch_size = 1

    def run(self):
        handles = list()
        for i in range(self.proc_num):
            handles.append(self.pool.apply_async(self.dummy_splitter, args=(i,)))

        results = []
        for r in handles:
            results.extend(r.get())
        self.pool.close()
        self.pool.join()
        return results

    def dummy_splitter(self, n):
        if n == (self.proc_num - 1):
            curr_slice = slice(n * self.batch_size, len(self.items))
        else:
            curr_slice = slice(n * self.batch_size, (n + 1) * self.batch_size)
        if curr_slice.start >= len(self.items):
            return []
        sub_items = self.items[curr_slice]
        if self.show_tqdm:
            return [self.function(item) for item in tqdm(sub_items)]
        return [self.function(item) for item in sub_items]

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict
