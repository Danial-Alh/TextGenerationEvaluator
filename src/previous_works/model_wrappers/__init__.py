# from previous_works.model_wrappers.others import DGSAN, Real, TextGan, LeakGan
from types import SimpleNamespace
from previous_works.model_wrappers.texygen import TexyGen
from previous_works.model_wrappers.rvae import VAE


def create_model(model_identifier: SimpleNamespace, parser):
    model_name = model_identifier.model_name.lower()
    model_class = model_name_class_mapping[model_name]
    if model_class == TexyGen:
        m = model_class(model_name, model_identifier, parser)
    else:
        m = model_class(model_identifier, parser)
    return m


model_name_class_mapping = {
    'seqgan': TexyGen,
    'rankgan': TexyGen, 'maligan': TexyGen, 'mle': TexyGen,
    'vae': VAE
}
# model_name_class_mapping = {
#     'dgsan': DGSAN,
#     'leakgan': LeakGan, 'textgan': TextGan, 'seqgan': TexyGen,
#     'rankgan': TexyGen, 'maligan': TexyGen, 'mle': TexyGen,
#     'vae': VAE,
#     'real': Real
# }
all_model_names = model_name_class_mapping.keys()
