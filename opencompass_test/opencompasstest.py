from opencompass.models import MydeepseekmoeModel
from opencompass.models import MyxversemoeModel
from opencompass.models import MyqwenmoeModel
from mmengine.config import read_base

# Pre-compute paths as plain strings via inline __import__ to avoid
# binding the os module into the mmengine-serialized config namespace.
_MODEL_BASE    = __import__('os').environ.get('MODEL_BASE', '/mnt/data/zgy')
_PATH_DEEPSEEK = _MODEL_BASE + '/deepseekmoe'
_PATH_XVERSE   = _MODEL_BASE + '/xversemoe'
_PATH_QWEN     = _MODEL_BASE + '/qwen2_moe'

with read_base():
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets
    from opencompass.configs.datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets

    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_314797 import BoolQ_datasets
    from opencompass.configs.datasets.race.race_gen_69ee4f import race_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets

    from opencompass.configs.datasets.flores.flores_gen_806ede import flores_datasets
    from opencompass.configs.datasets.triviaqa.triviaqa_gen_2121ce import \
        triviaqa_datasets


models = [
    dict(
        type=MydeepseekmoeModel,
        path=_PATH_DEEPSEEK,
        model_kwargs=dict(device_map="cuda:0"),
        tokenizer_path=_PATH_DEEPSEEK,
        max_seq_len=20480,
        max_out_len=100,
        run_cfg=dict(num_gpus=1, num_procs=1),
        batch_size=1,
    ),
    dict(
        type=MyxversemoeModel,
        path=_PATH_XVERSE,
        model_kwargs=dict(device_map="cuda:0"),
        tokenizer_path=_PATH_XVERSE,
        max_seq_len=20480,
        max_out_len=100,
        run_cfg=dict(num_gpus=1, num_procs=1),
        batch_size=1,
    ),
    dict(
        type=MyqwenmoeModel,
        path=_PATH_QWEN,
        model_kwargs=dict(device_map="cuda:0"),
        tokenizer_path=_PATH_QWEN,
        max_seq_len=20480,
        max_out_len=100,
        run_cfg=dict(num_gpus=1, num_procs=1),
        batch_size=1,
    ),
]

models[0]['generation_kwargs'] = dict(do_sample=False)
gsm8k_datasets[0]['reader_cfg']['test_range'] = '[0:1000]'
GaokaoBench_datasets = [GaokaoBench_datasets[0], GaokaoBench_datasets[1],
                        GaokaoBench_datasets[2], GaokaoBench_datasets[3]]
WiC_datasets[0]['reader_cfg']['test_range'] = '[0:1000]'
triviaqa_datasets[0]['reader_cfg']['test_range'] = '[0:1000]'
race_datasets[0]['reader_cfg']['test_range'] = '[0:1000]'
race_datasets[1]['reader_cfg']['test_range'] = '[0:0]'

datasets = GaokaoBench_datasets + WiC_datasets + triviaqa_datasets + race_datasets + gsm8k_datasets
