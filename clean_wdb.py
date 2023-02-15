import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *
import wandb

# TODO: need to do this occassionally
api = wandb.Api(overrides={"project": 'siqa', "entity": 'socialiq'})
for run in tqdm(api.runs()):
    if 'np1' in run.tags or 'debug' in run.tags:
    # if True:
        run.delete()
exit()