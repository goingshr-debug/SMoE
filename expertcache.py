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
ExpertUID = Tuple[int,int]

import os
import logging
logger = logging.getLogger(__name__)

def current_thread_affinity_load():
    pid = os.getpid()
    # 获取当前线程ID，注意这个 ID 可能在不同的系统上表示不同
    thread_id = threading.get_native_id()
    # 获取该线程的亲和性
    affinity = os.sched_getaffinity(pid)
    logger.debug(f"Current Thread ID for loading: {thread_id}")
    logger.debug(f"Thread can run on CPUs for loading: {affinity}")


@dataclass(frozen=False)
class ExpertInfo:
    uid: ExpertUID
    offloaded: bool
    priority: int
    loading: bool
    index: int


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

    def choose_expert_to_evict(self) -> ExpertInfo:
        s = time.time()
        min_priority = min(info.priority for info in self.main_infos.values())
        assert min_priority <2, "Cache size is too small to support normal system operation."
        for uid, info in self.main_infos.items():
            if info.priority == min_priority:
                e = time.time()
                logger.debug("choose_expert_to_evicttime  %s",e-s)
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
    def __init__(self, make_module_cuda: callable, make_module_cpu: callable, main_size: int, offload_size: int,window_size, state_dict_00):
        """Dynamically loads an array of modules with identical hyperparameters"""
        self.module_type = self.module_size = self.device = None
        self.active = False

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()
        self.main_modules = []
        # import psutil
        for _ in tqdm(range(main_size),desc = "init cache space for experts in GPU memory"):
            self.main_modules.append(self._check_module(make_module_cuda("/home/guoying/ourwork/models/deepseekmoe/model_path/model_params","deepseekmoe","cuda:1",state_dict_00))) #TODO
        logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        self.main_infos = [0 for _ in range(main_size)]

        assert self.module_size is not None
        self.offloaded_storages = []
        for _ in tqdm(range(offload_size),desc = "init offloading space in CPU memory"):
            self.offloaded_storages.append(make_module_cpu("/home/guoying/ourwork/models/deepseekmoe/model_path/model_params","deepseekmoe","cuda:1",state_dict_00))
        logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        self.offloaded_infos = [0 for _ in range(offload_size)]

        self.cache_infos = EvictionInfo()
        self.load_queue = deque()
        self.capacity_1 = main_size//2
        self.priority_one_queue = deque()
        self.mtx = threading.Lock()
        self.cv = threading.Condition(self.mtx)
        self.loading_thread = threading.Thread(target=self.loading,daemon=True)
        self.loading_thread.start()
        self.load_stream = torch.cuda.Stream()
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
                    info = ExpertInfo(uid, False,0,False, index=i)
                    self.registered_experts[uid] = info
                    self.cache_infos.add(info)
                    self.main_infos[i] =1
                    return  # done allocating; found spot on device
        if offload is None or offload:  # True or None
            for i in range(len(self.offloaded_storages)):
                if self.offloaded_infos[i] == 0:
                    self.offloaded_storages[i].storage.copy_(storage)
                    info = ExpertInfo(uid, True, 0,False,index=i)
                    self.registered_experts[uid] = info
                    self.cache_infos.add(info)
                    self.offloaded_infos[i] = 1
                    return  # done allocating; found an offloaded spot
        raise ValueError("Cache is full")


    def _swap(self, info_to_load_index, info_to_evict_index):
        # swap a single on-device expert with a single offloaded expert using buffers for parallelism
        start = time.time()
        # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        offloaded_storage_buffer = torch.UntypedStorage(self.module_size)
        offloaded_storage_buffer.copy_(self.main_modules[info_to_evict_index].storage,non_blocking=True)
        self.main_modules[info_to_evict_index].storage.copy_(self.offloaded_storages[info_to_load_index].storage,non_blocking=True)
        self.offloaded_storages[info_to_load_index].storage.copy_(offloaded_storage_buffer,non_blocking=True)
            # logger.debug(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        del offloaded_storage_buffer
        # torch.cuda.synchronize()
        # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        end = time.time()
        if len(self.LoadTimeOneExpert)<10:
            self.LoadTimeOneExpert.append( end-start)
        else:
            self.LoadTimeOneExpert.append( end-start)
            self.LoadTimeOneExpert = self.LoadTimeOneExpert[-10:]

    def get_compute_expert(self,uid:ExpertUID,offload=False):
        info = self.registered_experts[uid]
        if not offload:
            return self.main_modules[info.index]
        else:
            return self.offloaded_storages[info.index]

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
    def loading(self):
        # os.sched_setaffinity(0, {63})
        while True:
            with self.cv:
                # current_thread_affinity_load()
                self.cv.wait_for(lambda :len(self.load_queue)>0)
                uid = self.load_queue.popleft()
                if self.query_expert(uid):
                    return
                logger.debug("begin to load  %s",uid)
                assert uid in self.cache_infos.offloaded_infos
                self.registered_experts[uid].loading = True
                info_to_load = self.registered_experts[uid]
                info_to_evict = self.cache_infos.choose_expert_to_evict()
                assert info_to_load.offloaded and not info_to_evict.offloaded
                self.registered_experts[info_to_evict.uid].offloaded = True

                evictindex = info_to_evict.index
                loadindex = info_to_load.index
                self.registered_experts[uid].index = evictindex
                self.registered_experts[info_to_evict.uid].index = loadindex
                self._swap(loadindex,evictindex)

                self.registered_experts[uid].loading=False
                self.registered_experts[uid].offloaded=False
                logger.debug("loading time  %s",uid,self.LoadTimeOneExpert[-1])
                self.cache_infos.swap(info_to_load,info_to_evict)
                assert uid in self.cache_infos.main_infos
                self.cv.notify_all()
    def add_to_queue(self,uid:ExpertUID,predict:bool):
        with self.cv:
            if predict:
                if uid not in self.load_queue and not self.registered_experts[uid].loading:
                    self.load_queue.append(uid)
                    self.cv.notify_all()
            else:
                if not self.registered_experts[uid].loading:
                    if uid in self.load_queue:
                        self.load_queue.remove(uid)
                        self.load_queue.appendleft(uid)
                        self.cv.notify_all()
                    else:
                        self.load_queue.appendleft(uid)
                        self.cv.notify_all()
    def wait_until_queue_empty(self):
        with self.cv:
            while len(self.load_queue) > 0:
                self.cv.wait()
def prefill_replaceset_between_tokens(scores:list,a:float,topk):
    replaceset = set()
    n_tokens = len(scores)
    n_experts = len(scores[0])
    sort_index = [sorted(range(len(input_list)),key=lambda i:input_list[i],reverse=True) for input_list in scores]
    sort_scores = [[scores[j][i] for i in sort_index[j]] for j in range(n_tokens)]
    for token_id in range(n_tokens):
        midscore = sort_scores[token_id][topk]
        lscore = midscore +a*midscore
        for expert_i in range(n_experts):
            if sort_scores[token_id][expert_i] > lscore:
                replaceset.add(sort_index[token_id][expert_i])
    return list(replaceset)

def cache_router(scores:list,cache:ExpertCache,a:float,topk:int,replaceset:list,layer_id:int):
    n_tokens =len(scores)
    n_experts = len(scores[0])
    cacherouter_experts = [[None for i in range(topk)] for j in range(n_tokens)]
    top_uid = [[] for j in range(n_tokens)]
    sort_index = [sorted(range(len(input_list)),key=lambda i:input_list[i],reverse=True) for input_list in scores]
    sort_scores = [[scores[j][i] for i in sort_index[j]] for j in range(n_tokens)]
    for token_id in range(n_tokens):
        midscore = sort_scores[token_id][topk]
        lscore = midscore+a*midscore
        rscore = midscore-a*midscore
        high_num=0
        canreplaceset = set()
        showexperts = []
        for expert_i in range(n_experts):
            expertid = sort_index[token_id][expert_i]
            if sort_scores[token_id][expert_i]>lscore:
                cache.ready_compute((layer_id,expertid))
                cacherouter_experts[token_id][expert_i] =expertid
                top_uid[token_id].append((layer_id,expertid))
                high_num+=1
            elif rscore < sort_scores[token_id][expert_i] < midscore:
                canreplaceset.add(expertid)

        for expert_i in range(high_num,topk):
            expertid = sort_index[token_id][expert_i]
            uid = (layer_id,expertid)
            if cache.query_expert(uid) or expertid in replaceset:
                cacherouter_experts[token_id][expert_i] =expertid
                cache.ready_compute((layer_id,expertid))
                continue
            flag = 0
            for replaceexpertid in canreplaceset:
                replaceuid = (layer_id,replaceexpertid)
                if (cache.query_expert(replaceuid) or replaceexpertid in replaceset) and replaceexpertid not in cacherouter_experts[token_id]:
                    cacherouter_experts[token_id][expert_i] = replaceexpertid
                    canreplaceset.remove(replaceexpertid)
                    cache.ready_compute(replaceuid)
                    flag=1
                    break
            if flag==1:
                continue
            else:
                cacherouter_experts[token_id][expert_i]=expertid
                cache.ready_compute(uid)
        logger.debug("low  %s",cacherouter_experts)
    return cacherouter_experts,top_uid
