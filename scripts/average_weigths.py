import torch

base_model = 'models/efold_V1_PT+FT_epoch.pt'
base_model = base_model.split(".pt")[0]
epochs = [5,10]
models = [f'{base_model}{n}.pt' for n in range(epochs[0], epochs[1]+1)]

final_model = torch.load(models[0], map_location=torch.device('cpu'))
for model in models[1:]:
    new_model = torch.load(model, map_location=torch.device('cpu'))

    for key in new_model:
        final_model[key] += new_model[key]

for key in final_model:
    final_model[key] = (final_model[key] / len(models)).to(final_model[key].dtype)

torch.save(final_model, f'{base_model}{epochs[0]}-{epochs[1]}_avg.pt')