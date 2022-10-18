import torch.nn as nn

from .fairseq_dropout import FairseqDropout


def get_language_dropout_rate(lang):
    dropout_rate_for_each_language = {
        "bos": 0.4,
        "mar": 0.4,
        "hin": 0.4,
        "mkd": 0.4,
        "ell": 0.4,
        "bul": 0.4,
        "fra": 0.4,
        "kor": 0.4,
    }
    return dropout_rate_for_each_language[lang]


class LSDropout(nn.Module):
    """
    Implementation of Language-Specific Dropout
    """

    def __init__(self, args, module_name=None):
        super().__init__()
        # get all IDs of src tokens and target tokens in the dict
        # print("eachan print lang_pairs:", args.lang_pairs)
        self.args = args

        # create a dropout for each language
        self.dropout_dict = nn.ModuleDict()
        # print(args)
        for langpair in args.lang_pairs.split(','):
            if args.encoder_langtok == "src":
                lang = langpair.split("-")[0]
            elif args.encoder_langtok == "tgt":
                lang = langpair.split("-")[1]

            lang_token = '__{}__'.format(lang)
            lang_id = args.lang2id[lang_token]

            # 我这里简单点弄（我累了，应该不是UNK
            assert lang_id != 3

            # get the dropout rate of this language:
            p = get_language_dropout_rate(lang)
            self.dropout_dict[str(lang_id)] = FairseqDropout(p, module_name=module_name)

    def forward(self, langs, x, inplace: bool = False):
        assert len(langs.unique()) == 1
        lang_id = str(langs[0].item())
        return self.dropout_dict[lang_id](x, inplace)

