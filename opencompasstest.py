
from opencompass.models import MydeepseekmoeModel
from opencompass.models import MyxversemoeModel
from opencompass.models import MyqwenmoeModel
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets
    from opencompass.configs.datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_314797 import BoolQ_datasets
    from opencompass.configs.datasets.race.race_gen_69ee4f import race_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
        
    from opencompass.configs.datasets.ARC_c.ARC_c_gen_1e0de5 import \
        ARC_c_datasets
        
    from opencompass.configs.datasets.flores.flores_gen_806ede import flores_datasets
    from opencompass.configs.datasets.triviaqa.triviaqa_gen_2121ce import \
        triviaqa_datasets

    # from ...GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets

    # from opencompass.configs.datasets.hellaswag.hellaswag_ppl_a6e128 import hellaswag_datasets
    # from opencompass.configs.datasets.lambada.lambada_gen_217e11 import lambada_datasets
    # from opencompass.configs.datasets.Xsum.Xsum_gen_31397e import Xsum_datasets
    # from opencompass.configs.datasets.lcsts.lcsts_gen_8ee1fe import lcsts_datasets
    # from opencompass.configs.datasets.FewCLUE_csl.FewCLUE_csl_ppl_841b62 import csl_datasets
    # from opencompass.configs.datasets.demo.demo_math_chat_gen import \
    #     math_datasets
    # from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import \
    #     humaneval_datasets
    # from opencompass.configs.datasets.mbpp.deprecated_mbpp_repeat10_gen_1e1056 import \
    #     mbpp_datasets
    # from opencompass.configs.datasets.mbpp.deprecated_sanitized_mbpp_repeat10_gen_1e1056 import \
        # sanitized_mbpp_datasets
    # from opencompass.configs.datasets.GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets
    # from opencompass.configs.datasets.lambada.lambada_gen_217e11 import lambada_datasets
    # from ..Xsum.Xsum_gen_31397e import Xsum_datasets
    # from opencompass.configs.datasets.piqa.piqa_ppl_0cfff2 import piqa_datasets



models = [
    # dict(
    #     type=MydeepseekmoeModel,
    #     path='/home/guoying/ourwork/models/deepseekmoe/model_path/model_params',
    #     model_kwargs=dict(device_map="cuda:0"),
    #     tokenizer_path='/home/guoying/ourwork/models/deepseekmoe/model_path/model_params',
    #     max_seq_len=20480,
    #     max_out_len=100,
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    #     batch_size=1,
    # ),
    # dict(
    #     type=MydeepseekmoeModel,
    #     path='/home/guoying/ourwork/models/deepseekmoe/model_path/model_params',
    #     model_kwargs=dict(device_map="cuda:0"),
    #     tokenizer_path='/home/guoying/ourwork/models/deepseekmoe/model_path/model_params',
    #     max_seq_len=20480,
    #     max_out_len=100,
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    #     batch_size=1,
    # ),
    # dict(
    #     type=MyxversemoeModel,
    #     path='/home/guoying/ourwork/models/xversemoe/model_path',
    #     model_kwargs=dict(device_map="cuda:0"),
    #     tokenizer_path='/home/guoying/ourwork/models/xversemoe/model_path',
    #     max_seq_len=20480,
    #     max_out_len=100,
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    #     batch_size=1,
    # )
    dict(
        type=MyqwenmoeModel,
        path='/home/guoying/ourwork/models/qwemoe/model_path',
        model_kwargs=dict(device_map="cuda:0"),
        tokenizer_path='/home/guoying/ourwork/models/qwemoe/model_path',
        max_seq_len=20480,
        max_out_len=100,
        run_cfg=dict(num_gpus=1, num_procs=1),
        batch_size=1,
    )
]
# print(GaokaoBench_datasets)

models[0]['generation_kwargs'] = dict(do_sample=False)
# print(GaokaoBench_datasets)
BoolQ_datasets[0]['reader_cfg']['test_range'] = '[0:1000]'
gsm8k_datasets[0]['reader_cfg']['test_range'] = '[0:1000]'
# # newdatagaokao = GaokaoBench_datasets[0]
GaokaoBench_datasets=[GaokaoBench_datasets[0],GaokaoBench_datasets[1],GaokaoBench_datasets[2],GaokaoBench_datasets[3]]
# # ARC_c_datasets[0]['reader_cfg']['test_range'] = '[0:128]'
WiC_datasets[0]['reader_cfg']['test_range'] = '[0:1000]'
# # flores_datasets[0]['reader_cfg']['test_range'] = '[0:10]'
# # BoolQ_datasets[0]['reader_cfg']['test_range'] = '[0:128]'
triviaqa_datasets[0]['reader_cfg']['test_range'] = '[0:1000]'
race_datasets[0]['reader_cfg']['test_range'] = '[0:1000]'
race_datasets[1]['reader_cfg']['test_range'] = '[0:0]'
# hellaswag_datasets[0]['reader_cfg']['test_range'] = '[0:128]'
# lambada_datasets[0]['reader_cfg']['test_range'] = '[0:64]'
# Xsum_datasets[0]['reader_cfg']['test_range'] = '[0:64]'
# lcsts_datasets[0]['reader_cfg']['test_range'] = '[0:64]'
# csl_datasets[0]['reader_cfg']['test_range'] = '[0:64]'
# csl_datasets[1]['reader_cfg']['test_range'] = '[0:64]'
# math_datasets[0]['reader_cfg']['test_range'] = '[0:64]'
# humaneval_datasets[0]['reader_cfg']['test_range'] = '[0:64]'
# mbpp_datasets[0]['reader_cfg']['test_range'] = '[0:1]'
# sanitized_mbpp_datasets[0]['reader_cfg']['test_range'] = '[0:1]'
# datasets =GaokaoBench_datasets+WiC_datasets+triviaqa_datasets+race_datasets+gsm8k_datasets
# datasets =GaokaoBench_datasets+gsm8k_datasets
datasets=GaokaoBench_datasets+WiC_datasets+triviaqa_datasets+race_datasets+gsm8k_datasets
# datasets = flores_datasets


# GaokaoBench_datasets =  GaokaoBench_datasets[14:]
# datasets = GaokaoBench_datasets