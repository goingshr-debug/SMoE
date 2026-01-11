from dataclasses import dataclass,field
from typing import Dict, Optional, Tuple
from collections import deque, OrderedDict
import threading
import torch
from torch import nn
import copy
import time
from tqdm import tqdm
import psutil
import torch.profiler
import numpy as np
import scipy.stats as stats
ExpertUID = Tuple[int,int]
import logging
import os
from models.qwenmoe.model_path import modeling_qwen
#from models.deepseekmoe.model_path import modeling_deepseek
#from models.xversemoe.model_path import modeling_xverse
from streamer import StreamManager


logger = logging.getLogger(__name__)

# low_score_ratio=0
# router_time =0 
pcie_load_time=0

class FixedSizeQueueForScore():
    def __init__(self, k):
        self.k = k 
        self.queue = deque() 
        self.sum = 0 

    def add(self, value):
        if len(self.queue) == self.k:
            oldest = self.queue.popleft()
            self.sum -= oldest
        self.queue.append(value)
        self.sum += value

    def get_average(self):
        if len(self.queue) == 0:
            return 0 

        return self.sum / len(self.queue)



@dataclass(frozen=False)
class ExpertInfo:
    uid: ExpertUID
    offloaded: bool
    priority: int
    loading: bool
    scores: FixedSizeQueueForScore
    index: int
    offload_index:int


@dataclass
class EvictionInfo():
    # infos in main and offload devices; ordered from least recently used to most
    main_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    offloaded_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    hits: int = field(default=0)
    misses: int = field(default=0)

    def add(self, info: ExpertInfo):
        infos_odict = self.offloaded_infos if info.offloaded else self.main_infos
        assert info.uid not in infos_odict, f"expert {info.uid} already exists"
        infos_odict[info.uid] = info

    def choose_expert_to_evictbyScore(self) -> ExpertInfo:
        min_priority = float('inf')
        evict_info = None
        evict_average = float('inf')
        
        for uid, info in self.main_infos.items():
            if info.priority < min_priority:
                min_priority = info.priority
                evict_info = info
                evict_average = info.scores.get_average()
            elif info.priority == min_priority:
                current_average = info.scores.get_average()
                if current_average < evict_average:
                    evict_info = info
                    evict_average = current_average

        assert min_priority < 2, "Cache size is too small to support normal system operation."
        
        if evict_info is None:
            raise ValueError("No evictable experts")

        return evict_info

    def choose_expert_to_evictbyLRU(self):
        min_priority = min(info.priority for info in self.main_infos.values())
        assert min_priority <2, "Cache size is too small to support normal system operation."
        for uid, info in self.main_infos.items():
            if info.priority == min_priority:
                return info  # least recently used
        raise ValueError("No evictable experts")

    def swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo):
        assert info_to_load.uid in self.offloaded_infos and info_to_evict.uid in self.main_infos
        self.main_infos[info_to_load.uid] = self.offloaded_infos.pop(info_to_load.uid)
        self.main_infos.move_to_end(info_to_load.uid, last=True)
        self.offloaded_infos[info_to_evict.uid] = self.main_infos.pop(info_to_evict.uid)

    def mark_pro(self,info: ExpertInfo,priority:int):
        assert info.uid in self.main_infos
        self.main_infos[info.uid].priority = priority
    def mark_used(self, info: ExpertInfo):
        if info.uid in self.main_infos:
            self.main_infos.move_to_end(info.uid, last=True)
            self.hits += 1
        elif info.uid in self.offloaded_infos:
            self.offloaded_infos.move_to_end(info.uid, last=True)
            self.misses += 1
        else:
            raise ValueError(f"Expert {info} not in group")


class ExpertCache:
    def __init__(self, config,make_module_cuda: callable, make_module_cpu: callable, main_size: int, offload_size: int,window_size:int ,state_dict_00,model_type,model_path):
        """Dynamically loads an array of modules with identical hyperparameters"""
        self.module_type = self.module_size = self.device = None
        self.active = False

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()
        self.main_modules = []
        # import psutil
        for _ in tqdm(range(main_size),desc = "init cache space for experts in GPU memory"):
            self.main_modules.append(self._check_module(make_module_cuda(model_path,model_type,config.device,state_dict_00)))
        logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        self.main_infos = [0 for _ in range(main_size)]

        assert self.module_size is not None
        self.offloaded_storages = []
        for _ in tqdm(range(offload_size),desc = "init offloading space in CPU memory"):
            self.offloaded_storages.append(make_module_cpu(model_path,model_type,config.device,state_dict_00))
        logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        self.offloaded_infos = [0 for _ in range(offload_size)]
        self.cache_window = window_size
        self.cache_infos = EvictionInfo()
        
        self.load_queue = deque()
        self.capacity_1 = main_size//2
        self.priority_one_queue = deque()
        self.mtx = threading.Lock()
        self.cv = threading.Condition(self.mtx)
        self.loading_thread = threading.Thread(target=self.loading,daemon=True)
        self.loading_thread.start()
        self.load_stream = torch.npu.Stream(device=config.device)
        self.LoadTimeOneExpert = [0.05]
    def _check_module(self, module: nn.Module):
        assert isinstance(module.storage, torch.UntypedStorage)
        if self.module_type is None:
            self.module_type = type(module)
            self.module_size = len(module.storage)
            self.device = module.storage.device
        else:
            assert isinstance(module, self.module_type)
            assert len(module.storage) == self.module_size
            assert module.storage.device == self.device
        return module
    def query_expert(self,uid: ExpertUID):
        with self.mtx:
            if uid in self.cache_infos.main_infos and not self.cache_infos.main_infos[uid].offloaded:
                return True
            return False
    def query_expert_inload(self,uid: ExpertUID):
        if uid in self.cache_infos.main_infos and not self.cache_infos.main_infos[uid].offloaded:
            return True
        return False
    def add_expert(self, uid: ExpertUID, module: nn.Module, offload: Optional[bool] = None):
        # with self.mtx:
        assert self.module_type is not None
        assert isinstance(module, self.module_type)
        return self.add_expert_storage(uid, module.storage, offload=offload)

    def add_expert_storage(self, uid: ExpertUID, storage: torch.UntypedStorage, offload: Optional[bool] = None):
        assert uid not in self.registered_experts, f"expert {uid} already registered"
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size

        if offload is None or not offload:  # False or None
            for i in range(len(self.main_modules)):
                if self.main_infos[i] == 0:
                    # logger.debug("----------")
                    # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                    self.main_modules[i].storage.copy_(storage)
                    # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                    info = ExpertInfo(uid, False,0,False, scores = FixedSizeQueueForScore(self.cache_window),index=i,offload_index=0)
                    self.registered_experts[uid] = info
                    self.cache_infos.add(info)
                    self.main_infos[i] =1
                    break
        for i in range(len(self.offloaded_storages)):
            if self.offloaded_infos[i] == 0:
                self.offloaded_storages[i].storage.copy_(storage)
                if offload:
                    info = ExpertInfo(uid, True, 0,False,scores = FixedSizeQueueForScore(self.cache_window),index=i,offload_index=i)
                    self.registered_experts[uid] = info
                    self.cache_infos.add(info)
                else:
                    self.registered_experts[uid].offload_index=i
                self.offloaded_infos[i] = 1
                return  # done allocating; found an offloaded spot
        raise ValueError("Cache is full")


    def _swap(self, info_to_load_index, info_to_evict_index):

        start = time.time()
        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=True) as prof:
            # with torch.profiler.record_function("COPY_OP"):
        with torch.npu.stream(self.load_stream):
            self.main_modules[info_to_evict_index].storage.copy_(self.offloaded_storages[info_to_load_index].storage)
        # self.offloaded_storages[info_to_load_index].storage.copy_(offloaded_storage_buffer)
            # logger.debug(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        # del offloaded_storage_buffer
        # torch.cuda.synchronize()
        # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        torch.npu.synchronize()
        end = time.time()
        # logger.info('experts swap %s', end-start)
        # logger.debug(prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=10))
        if len(self.LoadTimeOneExpert)<10:
            self.LoadTimeOneExpert.append( end-start)
        else:
            self.LoadTimeOneExpert.append( end-start)
            self.LoadTimeOneExpert = self.LoadTimeOneExpert[-10:]

    def get_compute_expert(self,uid:ExpertUID,offload=False):
        with self.mtx:
            info = self.registered_experts[uid]
            if not offload:
                # logger.debug("self.main_modules[info.index]",uid,info.index)
                return self.main_modules[info.index]
            else:
                return self.offloaded_storages[info.offload_index]

    def ready_compute(self,uid:ExpertUID):
        self.registered_experts[uid].priority=2
        if uid in self.priority_one_queue:
            self.priority_one_queue.remove(uid)

    def predict_compute(self,uid: ExpertUID):
        if self.registered_experts[uid].priority==2:
            return
        self.registered_experts[uid].priority=1
        self.priority_one_queue.append(uid)
        self._check_priority_capacity()
    def end_compute(self,uid:ExpertUID):
        self.registered_experts[uid].priority=0
        if uid in self.priority_one_queue:
            self.priority_one_queue.remove(uid)
    def _check_priority_capacity(self):
        while len(self.priority_one_queue)>self.capacity_1:
            oldest_uid = self.priority_one_queue.popleft()
            if oldest_uid in self.registered_experts:
                self.registered_experts[oldest_uid].priority = 0
    def update_scores(self,layerid,scores_lst):
        for expertid in range(64):
            uid = (layerid,expertid)
            score = scores_lst[expertid]
            self.registered_experts[uid].scores.add(score)
    def loading(self):
        # os.sched_setaffinity(0, {63})
        while True:
            with self.cv:
                # current_thread_affinity_load()
                self.cv.wait_for(lambda :len(self.load_queue)>0)
                uid = self.load_queue[0]
                if self.query_expert_inload(uid):
                    # logger.debug("popuid %s",uid)
                    self.load_queue.popleft()
                    continue
                # logger.debug("begin to load %s",uid)
                assert uid in self.cache_infos.offloaded_infos
                self.registered_experts[uid].loading = True
                info_to_load = self.registered_experts[uid]
                evict_start = time.time()
                info_to_evict = self.cache_infos.choose_expert_to_evictbyScore() if self.cache_window!=None else self.cache_infos.choose_expert_to_evictbyLRU()
                evict_end = time.time()
                # logger.info('cpu eviction time: %s', evict_end - evict_start)
                assert info_to_load.offloaded and not info_to_evict.offloaded
                self.registered_experts[info_to_evict.uid].offloaded = True

                evictindex = info_to_evict.index
                loadindex = info_to_load.offload_index
            self._swap(loadindex,evictindex)
            with self.cv:
                self.registered_experts[uid].index = evictindex
                self.registered_experts[info_to_evict.uid].index = info_to_evict.offload_index
                # logger.debug("index update self.registered_experts[uid].index %s",uid,evictindex)
                if len(self.load_queue) >0 and uid == self.load_queue[0]:
                    self.load_queue.popleft()
                self.registered_experts[uid].loading=False
                self.registered_experts[uid].offloaded=False
                # global pcie_load_time
                if modeling_qwen.tokens > 0: #only qwen2
                    modeling_qwen.pcie_load_time += self.LoadTimeOneExpert[-1]
                    # print("qwen loading time %s %s",uid,self.LoadTimeOneExpert[-1])
#                if modeling_deepseek.tokens > 0:
#                    modeling_deepseek.pcie_load_time += self.LoadTimeOneExpert[-1]
                    # print("deepseek loading time %s %s",uid,self.LoadTimeOneExpert[-1])
#                if modeling_xverse.tokens > 0:
#                    modeling_xverse.pcie_load_time += self.LoadTimeOneExpert[-1]

                    # print("xverse loading time %s %s",uid,self.LoadTimeOneExpert[-1])

                self.cache_infos.swap(info_to_load,info_to_evict)
                assert uid in self.cache_infos.main_infos
                self.cv.notify_all()
    def add_to_queue(self,uid:ExpertUID):
        with self.cv:
            if uid not in self.load_queue and not self.query_expert_inload(uid):
                
                self.load_queue.append(uid)
                self.cv.notify_all()
    def clear_queue(self):
        with self.cv:
            self.load_queue = deque()
    def wait_until_queue_empty(self):
        with self.cv:
            self.cv.wait_for(lambda :len(self.load_queue)==0)
def replaceset_between_tokens(scores:list,a:float,topk):
    replaceset = set()
    # allset = set()
    n_tokens = len(scores)
    n_experts = len(scores[0])
    sort_index = [sorted(range(len(input_list)),key=lambda i:input_list[i],reverse=True) for input_list in scores]
    sort_scores = [[scores[j][i] for i in sort_index[j]] for j in range(n_tokens)]
    for token_id in range(n_tokens):
        midscore = sort_scores[token_id][topk]
        # lscore = find_smallest_max_outlier(sort_scores[token_id][:topk+6])
        lscore = midscore+a*midscore
        for expert_i in range(n_experts):
            if sort_scores[token_id][expert_i] >= lscore and expert_i<topk:
                replaceset.add(sort_index[token_id][expert_i])
            # if sort_scores[token_id][expert_i] > midscore:
            #     allset.add(sort_index[token_id][expert_i])
    return list(replaceset)

def cache_router(scores:list,cache:ExpertCache,a:float,topk:int,replaceset:list,layer_id:int):
    n_tokens =len(scores)
    n_experts = len(scores[0])
    cacherouter_experts = [[None for i in range(topk)] for j in range(n_tokens)]
    top_uid = [[] for j in range(n_tokens)]
    sort_index = [sorted(range(len(input_list)),key=lambda i:input_list[i],reverse=True) for input_list in scores]
    sort_scores = [[scores[j][i] for i in sort_index[j]] for j in range(n_tokens)]
    expertdic_batch = dict()
    tokendict_alter = dict()
    tokendict_highnum = dict()
    
    #global low_score_ratio
    #global router_time  
    for token_id in range(n_tokens):
        highscoreneedload=0
        midscore = sort_scores[token_id][topk]
        # lscore = find_smallest_max_outlier(sort_scores[token_id][:topk+6])
        lscore = midscore+a*midscore
        logger.debug(sort_scores[token_id][:topk])
        logger.debug(lscore)
        shapiro_test = stats.shapiro(sort_scores[token_id])
        # logger.debug(f"Shapiro-Wilk Test: p-value = {shapiro_test.pvalue}")
        rscore = midscore-a*midscore
        high_num=0
        canreplaceset = set()
        for expert_i in range(n_experts):
            expertid = sort_index[token_id][expert_i]
            if sort_scores[token_id][expert_i]>=lscore and expert_i<topk:
                logger.debug(expert_i)
                cache.ready_compute((layer_id,expertid))
                cacherouter_experts[token_id][expert_i] =expertid
                top_uid[token_id].append((layer_id,expertid))
                high_num+=1
                uid = (layer_id,expertid)
                expertdic_batch[uid]=expertdic_batch.get(uid, 0) + 1
            #    if not cache.query_expert(uid):
            #        highscoreneedload+=1
            elif rscore < sort_scores[token_id][expert_i] < midscore:
                canreplaceset.add(expertid)
                tokendict_alter[token_id] = tokendict_alter.get(token_id,[])
                replaceuid = (layer_id,expertid)
                tokendict_alter[token_id].append(replaceuid)
                # print("expertid",expertid)
        
        low_score_experts_needload=0
        tokendict_highnum[token_id] = high_num
        # logger.debug("lowexpertoccupy")
        # logger.debug("top %s",cacherouter_experts)
        for expert_i in range(high_num,topk):
            expertid = sort_index[token_id][expert_i]
            uid = (layer_id,expertid)
            if cache.query_expert(uid) or expertid in replaceset:
                cacherouter_experts[token_id][expert_i] =expertid
                cache.ready_compute((layer_id,expertid))
                expertdic_batch[uid]=expertdic_batch.get(uid, 0) + 1
                continue
            flag = 0
            low_score_experts_needload+=1
            for replaceexpertid in canreplaceset:
                replaceuid = (layer_id,replaceexpertid)
                if (cache.query_expert(replaceuid) or replaceexpertid in replaceset) and replaceexpertid not in cacherouter_experts[token_id]:
                    cacherouter_experts[token_id][expert_i] = replaceexpertid
                    expertdic_batch[replaceuid]=expertdic_batch.get(replaceuid, 0) + 1
                    canreplaceset.remove(replaceexpertid)
                    cache.ready_compute(replaceuid)
                    flag=1
                    break
            if flag==1:
                continue
            else:
                cacherouter_experts[token_id][expert_i]=expertid
                expertdic_batch[uid]=expertdic_batch.get(uid, 0) + 1
                cache.ready_compute(uid)

    return cacherouter_experts,top_uid
def remove_outliers_and_average(raw):
    numbers = raw[:]
    if len(numbers) == 0:
        raise ValueError("列表不能为空。")
    
    # 如果列表中只有一个或两个数值，直接计算平均值并返回
    if len(numbers) <= 2:
        return np.mean(numbers)
    
    # 计算平均值和标准差
    mean = np.mean(numbers)
    std_dev = np.std(numbers)
    
    # 找出与平均值相差超过标准差的数值
    outliers = [x for x in numbers if abs(x - mean) > std_dev]
    
    # 从列表中移除这些数值
    filtered_numbers = [x for x in numbers if x not in outliers]
    
    # 如果移除异常值后没有剩余数值，直接返回原始列表的平均值
    if len(filtered_numbers) == 0:
        return np.mean(numbers)
    
    # 计算剩余数值的平均值
    average = np.mean(filtered_numbers)
    
    return average

def CPU_load_management(uid_batch,cpucost,loadcost):
    all_cpucost = 0
    all_loadcost =0
    uids = list(uid_batch.keys())
    l=0
    r=len(uids)-1
    uids = sorted(uids, key=lambda x: uid_batch[x], reverse=True)
    cpulst=[]
    loadlst=[]
    while l<=r:
        if all_loadcost<=all_cpucost:
            all_loadcost+=loadcost
            loadlst.append(uids[l])
            l+=1
        else:
            # all_cpucost+=(cpucost*uid_batch[uids[r]])
            all_cpucost+=(cpucost)
            cpulst.append(uids[r])
            r-=1
    
    return loadlst,cpulst

            
import numpy as np

def find_smallest_max_outlier(data, threshold=3):
    """
    找出最大异常值中最小的那个。
    
    参数：
    data (array-like): 一组数据。
    threshold (float): z-score阈值，用于判断是否为异常值，默认值为3。
    
    返回：
    float or None: 最大异常值中最小的那个值，如果无异常值，则返回 None。
    """
    # 计算均值和标准差
    data = np.asarray(data)
    mean = np.mean(data)
    std_dev = np.std(data)

    # 计算 z-score
    z_scores = (data - mean) / std_dev

    # 找出所有异常值
    outliers = data[np.abs(z_scores) > threshold]

    if outliers.size > 0:
        # 获取最大异常值中的最小值
        min_of_max_outliers = np.min(outliers)
        return min_of_max_outliers
    else:
        return mean

# 使用示例