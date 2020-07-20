# from previous_works.model_wrappers.others import DGSAN, Real, TextGan, LeakGan
from previous_works.model_wrappers.texygen import TexyGen
from previous_works.model_wrappers.rvae import VAE


def create_model(model_name, parser):
    model_name = model_name.lower()
    model_class = model_name_class_mapping[model_name]
    if model_class == TexyGen:
        m = model_class(model_name, parser)
    else:
        m = model_class(parser)
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
all_models = model_name_class_mapping.keys()
