def make_caption(dataset_name):
    caption = "Performance of models (using different measures) on \\textit{%s} dataset." % (dataset_name)
    # caption = "Models tarined on \"%s\" data. training stop criterion based on %s." % (good_dataset_name, suffix)
    return caption


class TblGenerator:
    def __init__(self, headers, caption, size, label, column_pattern=None):
        self.header_len = len(headers)
        if column_pattern is None:
            column_pattern_c = " ".join(["c"] * (self.header_len - 1))
        else:
            assert sum(column_pattern) < self.header_len
            new_column_pattern = column_pattern[:]
            if sum(column_pattern) < self.header_len - 1:
                new_column_pattern += self.header_len - sum(column_pattern) - 1
            column_pattern_c = "|".join([" ".join(["c"] * x) for x in column_pattern])

        self.data = "\\begin{table*}[!htb]\n\\centering\n\\caption{%s}\\label{%s}" % (caption, label)
        self.data += "\n\\%s\\tabcolsep=0.07cm\n\\begin{tabular}{||c||%s||}" % (size, column_pattern_c)
        # self.data += " \\tiny "
        self.data += "\\hline\\hline " + "\t& ".join(headers)
        # self.data += " \\normalsize "
        self.data += "\\\\\n\\hline\\hline\n"

    def add_row(self, data):
        assert len(data) == self.header_len
        tmp = "\t& ".join(data)
        tmp += " \\\\\n\\hline\n"
        self.data += tmp

    def __str__(self):
        tmp = self.data
        tmp += "\\hline\\end{tabular}\\normalsize \n \\end{table*}\n\n"
        return tmp


def change_metric_name(inp):
    metric_name_dict = {
        "embd": "W2BD",
        "nll": "NLL",
        "fbd": "FBD",
        "jaccard2": "MSJ-2",
        "jaccard3": "MSJ-3",
        "jaccard4": "MSJ-4",
        "jaccard5": "MSJ-5",
        "bleu2": "BL-2",
        "bleu3": "BL-3",
        "bleu4": "BL-4",
        "bleu5": "BL-5",
        "self_bleu2": "SBL-2",
        "self_bleu3": "SBL-3",
        "self_bleu4": "SBL-4",
        "self_bleu5": "SBL-5",

        "jeffreys": "Jeffreys",
        "bhattacharyya": "Bhattacharyya",
        "nllp_fromq": "OracleNLL",
        "nllq_fromp": "NLL",
        "nllp_fromp": None,
        "nllq_fromq": None
    }
    assert inp in metric_name_dict, inp
    return metric_name_dict[inp]
