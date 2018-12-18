class TblGenerator:
    def __init__(self, headers, caption, size, label):
        self.header_len = len(headers)
        self.data = "\\begin{table*}[!t]\n\\centering\n\\caption{%s}\\label{%s}" % (caption, label)
        self.data += "\n\\%s\\tabcolsep=0.11cm\n\\begin{tabular}{|c|%s|}" % (size, " ".join(
            ["c"] * (self.header_len - 1)))
        # self.data += " \\tiny "
        self.data += "\\hline " + "\t& ".join(headers)
        # self.data += " \\normalsize "
        self.data += "\\\\\n\\hline\n"

    def add_row(self, data):
        assert len(data) == self.header_len
        tmp = "\t& ".join(data)
        tmp += " \\\\\n\\hline\n"
        self.data += tmp

    def __str__(self):
        tmp = self.data
        tmp += "\\end{tabular}\\normalsize \n \\end{table*}\n\n"
        return tmp


def change_model_name(inp):
    model_name_dict1 = {"leakgan2": "leakgan",
                        "Maligan": "maligan",
                        "Mle": "mle",
                        "Rankgan": "rankgan",
                        "Seqgan": "seqgan"}
    model_name_dict2 = {"leakgan": "LeakGAN",
                        "maligan": "MaliGAN",
                        "mle": "MLE",
                        "rankgan": "RankGAN",
                        "seqgan": "SeqGAN"}
    if inp in model_name_dict1:
        inp = model_name_dict1[inp]
    return model_name_dict2[inp]


def change_metric_name(inp):
    metric_name_dict = {
        "-nll": "NLL",
        "jaccard2": "MSJ-2",
        "jaccard3": "MSJ-3",
        "jaccard4": "MSJ-4",
        "jaccard5": "MSJ-5",
        "bleu2": "BLEU-2",
        "bleu3": "BLEU-3",
        "bleu4": "BLEU-4",
        "bleu5": "BLEU-5",
        "self_bleu2": "SBLEU-2",
        "self_bleu3": "SBLEU-3",
        "self_bleu4": "SBLEU-4",
        "self_bleu5": "SBLEU-5",

        "jeffreys": "Jeffreys",
        "bhattacharyya": "Bhattacharyya",
        "lnp_fromq": "OracleNLL",
        "lnq_fromp": "NLL",
        "lnp_fromp": None,
        "lnq_fromq": None
    }
    return metric_name_dict[inp]
