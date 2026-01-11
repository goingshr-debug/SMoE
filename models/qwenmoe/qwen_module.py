import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
import random
from expertcachewithscorefullcpu import ExpertCache,cache_router,remove_outliers_and_average,replaceset_between_tokens,CPU_load_management
from typing import Tuple,List,OrderedDict
import time
import numpy as np
import threading
from .model_path import modeling_qwen
import logging
logger = logging.getLogger(__name__)

ExpertUID = Tuple[int,int]
# cache_ratio = 0
# times =0
# top_loads =0
# low_loads = 0
# tops =0
# lows=0
# scores_all =[0,0,0,0,0,0,0,0,0,0]
# scoretime=0
# compute_time_1=0
# CPUtime =0 



class Qwen2MoeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        # logger.debug("hid %s %s",self.hidden_size)
        # logger.debug("inter %s",self.intermediate_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False,dtype=torch.bfloat16, device=config.device)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False,dtype=torch.bfloat16, device=config.device)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False,dtype=torch.bfloat16, device=config.device)
        self.act_fn = ACT2FN[config.hidden_act]
        

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2MoeSparseMoeBlockwithCache(nn.Module):
    def __init__(self, config,expertcache: ExpertCache, layerid,gate:nn.Linear,shared_experts: Qwen2MoeMLP,shared_expert_gate:nn.Linear,next_attention,next_gate_weight,next_input_layernorm,next_post_attention_layernorm):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.ExpertCache = expertcache
        self.layerid = layerid
        self.replaceScoreRatio = config.replaceScoreRatio
        
        self.next_attention = next_attention
        self.next_gate_weight = next_gate_weight
        self.next_input_layernorm = next_input_layernorm
        self.next_post_attention_layernorm = next_post_attention_layernorm
        self.next_experts = None
        
        # gating
        self.gate = gate
        self.shared_expert = shared_experts
        self.shared_expert_gate = shared_expert_gate
        
        self.if_usecpu = config.if_usecpu
        self.if_prefetch = config.if_prefetch
        self.replaceScoreRatio = config.replaceScoreRatio
        self.config = config
        self.if_replace = config.if_replace
        
        
        self.CPUComputeTimeOneExpertOneBatch=[0.05]

    def forward(self, hidden_states: torch.Tensor,residual_cur,attn_weights_cur,present_key_value_cur,attention_mask,position_ids,output_attentions,cache_position,position_embeddings,) -> torch.Tensor:
        """ """
        global compute_time_1
        compute_time_1=0
        start = time.time()
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        s = routing_weights.tolist()
        
        
        sort_index = [sorted(range(len(input_list)),key=lambda i:input_list[i],reverse=True) for input_list in s]
        # logger.info  ('layer: %s, Scores: %s',self.layerid, sorted_s[:16])

        if self.ExpertCache.cache_window!=None:
            for score_l in routing_weights.tolist():
                self.ExpertCache.update_scores(self.layerid, score_l)
                
        topk_weight, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1, sorted=True)
        #TODO:for batchsize>1 when decoding.
        logger.debug(hidden_states.size()[0])
        replaceset = []
        if self.replaceScoreRatio!=None:

            if modeling_qwen.tokens > 0:
                replaceset = replaceset_between_tokens(routing_weights.tolist(),self.replaceScoreRatio,self.top_k)
                # print("replaceset",replaceset)
                # print("allset",allset)
                # print("top ,low",tops,lows)
                # logger.info('layer i selected_experts: % s, layer %s', topk_idx,self.layerid)
                # logger.info('layer i tops: %s, layer %s',replaceset,self.layerid)

                # if lows+tops > 0:
                #     logger.info("top ratio %s",tops/(lows+tops))
                # logger.debug("low ratio %s",lows/(lows+tops))
                if self.if_replace:
                    cacherouter_experts,_ = cache_router(routing_weights.tolist(),self.ExpertCache,self.replaceScoreRatio,self.top_k,replaceset,self.layerid)
                    topk_idx = torch.tensor(cacherouter_experts,device=topk_weight.device)
                    # logger.debug(cacherouter_experts)
                    topk_weight = routing_weights[torch.arange(routing_weights.size(0)).unsqueeze(1),topk_idx]
        for topl in topk_idx.tolist():
            for expert_id in topl:
                uid = (self.layerid,expert_id)
                self.ExpertCache.ready_compute(uid)
        if self.norm_topk_prob:
            topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(topk_idx, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        # if hidden_states.size()[0] == 1: 
        expert_token_dic=dict()
        topidxlst = []
        for l in topk_idx.tolist():
            for i in l:
                topidxlst.append(i)
        for expert_idx in range(self.num_experts):
            if expert_idx not in topidxlst:
                continue
            uid = (self.layerid,expert_idx)
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            expert_token_dic[uid] = [current_state,topk_weight[top_x, idx, None],top_x]
        #根据expert_token_dic和self.expertcache运算得到final_hidden_states
        
        attention_end = time.time()
        # logger.info('gate compute time: %s',attention_end - start)
        compute_time_1 += attention_end - start

        
        self.moe_infer_for_one_batch(replaceset,final_hidden_states,expert_token_dic,hidden_states.dtype,residual_cur,attn_weights_cur,present_key_value_cur,hidden_states,attention_mask,position_ids,output_attentions,cache_position,position_embeddings,(batch_size,sequence_length,hidden_dim))
        # else:
        #     expert_token_dic=dict()
        #     topidxlst = topk_idx.tolist()
        #     topidxlst = [item for sublist in topidxlst for item in sublist]
        #     for expert_idx in range(self.num_experts):
        #         if expert_idx not in topidxlst:
        #             continue
        #         uid = (self.layerid,expert_idx)
        #         idx, top_x = torch.where(expert_mask[expert_idx])
        #         current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        #         expert_token_dic[uid] = [current_state,topk_weight[top_x, idx, None],top_x]
        #     self.moe_infer(final_hidden_states,expert_token_dic,hidden_states.dtype)
        shared_start = time.time()
        for l in topk_idx.tolist():
            for i in l:
                uid = (self.layerid,i)
                self.ExpertCache.end_compute(uid)
        if modeling_qwen.tokens > 0:
            pass
            # logger.info("predict layer i+1 next_experts %s",self.next_experts)
            
        # if self.next_experts !=None:
        #     s= time.time()
        #     for experts_id in self.next_experts:
        #         uid = (self.layerid+1,experts_id)
        #         self.ExpertCache.add_to_queue(uid)
        #         logger.debug("prefetch %s",uid)
        #     e=time.time()
        #     logger.debug("loadtimeforexp %s",e-s)
        logger.debug("begincomputeshared")
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        
        end_forward = time.time()

        # logger.info('moe infer and shared compute: %s', end_forward - attention_end)
        compute_time_1 += end_forward - shared_start
        # logger.info("moeblock layer forward: %s", end_forward - start)
        # logger.info("compute time all: %s",compute_time_1)
        return final_hidden_states, router_logits
    
    def compute_expertsincache_in_thread(self,expertcomputeusecudaincache,x, expert_token_dic,hdtype,expert_num_not_in_cuda_num,residual_cur,attn_weights_cur,present_key_value_cur,identity,attention_mask,position_ids,output_attentions,expert_out_dict,cache_position,position_embeddings,bsh):
        # 这部分是在新线程中执行的for循环逻辑
        s= time.time()
        self.next_experts = None
        hidden_states = torch.zeros_like(x)
        for uid in expertcomputeusecudaincache:
            s1=time.time()
            expert = self.ExpertCache.get_compute_expert(uid)
            s2=time.time()
            logger.debug("getexperttime %s",s2-s1)
            expert_tokens = expert_token_dic[uid][0]
            weight = expert_token_dic[uid][1]
            top_x = expert_token_dic[uid][2]
            startcom = time.time()
            expert_out = expert(expert_tokens)
            endcom = time.time()
            expert_batch = expert_tokens.size()[0]
            # print("compute one expert time in GPU: %s %s", uid, endcom - startcom)
            # print("expert batch: %s",expert_batch)
            expert_out.mul_(weight)
            expert_out_dict[uid] = expert_out
            hidden_states.index_add_(0, top_x, expert_out.to(hdtype))
            s3 = time.time()
            logger.debug("last %s",s3-endcom)
            global compute_time_1
            compute_time_1 += s3 - startcom
            # logger.info('compute time cuda in new_thread %s', s3 - startcom)
            
        if self.layerid <27 and expert_num_not_in_cuda_num>0 and self.if_prefetch :
            s=time.time()
            self.next_experts = self.get_next_top_expert(residual_cur,attn_weights_cur,present_key_value_cur,hidden_states,identity,attention_mask,position_ids,output_attentions,cache_position,position_embeddings,bsh)
            e =time.time()
            compute_time_1 += e-s
            # logger.info("compute experts predict time: %s",e-s)
            # logger.info("next_experts %s",self.next_experts)
            


    @torch.no_grad()
    def moe_infer_for_one_batch(self, replaceset,final_hidden_states, expert_token_dic,hdtype,residual_cur,attn_weights_cur,present_key_value_cur,identity,attention_mask,position_ids,output_attentions,cache_position,position_embeddings,bsh):
        logger.debug("--------------")
        s= time.time()
        #分配哪些需要加载，哪些需要CPU计算,哪些专家可以直接cuda计算
        expertnotincudalst = []
        expertcomputeusecudaincache = []
        expert_num_not_in_cuda_num=0
        self.ExpertCache.clear_queue()
        self.ExpertCache.load_stream.synchronize()
        uid_batch = dict()
        newreplace=[]
        for i in replaceset:
            newreplace.append((self.layerid,i))
        for uid in expert_token_dic.keys():
            if not self.ExpertCache.query_expert(uid):
                expert_num_not_in_cuda_num+=1
                expertnotincudalst.append(uid)
                uid_batch[uid] = expert_token_dic[uid][0].size()[0]
                if modeling_qwen.tokens > 0:
                    if uid in newreplace:
                        modeling_qwen.top_loads+=1
                        # print("topload",uid)
                    else:
                        modeling_qwen.low_loads+=1
                        # print("lowload",uid)
                # logger.debug("lowloadratio %s",low_loads/all_loads)
                # logger.debug("toploadratio %s",top_loads/all_loads)
            else:
                expertcomputeusecudaincache.append(uid)
        # if modeling_qwen.tokens > 0:
        #     global tops,lows
        #     print("tops,lows",tops,lows)
        #     print("toplaod,lowload",top_loads,low_loads)
        #     if tops>0:
        #         logger.info("load_top_ratio %s",top_loads/tops)
        #     if lows>0:
        #         logger.info("load_low_ratio %s",low_loads/lows)
        CPUComputeTimeOneExpertOneBatch = remove_outliers_and_average(self.CPUComputeTimeOneExpertOneBatch)
        LoadTimeOneExpert = remove_outliers_and_average(self.ExpertCache.LoadTimeOneExpert) if len(self.CPUComputeTimeOneExpertOneBatch) >2 else CPUComputeTimeOneExpertOneBatch
        load_num = round(expert_num_not_in_cuda_num*(CPUComputeTimeOneExpertOneBatch/(CPUComputeTimeOneExpertOneBatch+LoadTimeOneExpert)))
        logger.debug('load_num: %s',load_num)
        logger.debug('expert_num_not_in_cuda_num: %s',expert_num_not_in_cuda_num)
        logger.debug('CPUComputeTimeOneExpertOneBatch: %s',CPUComputeTimeOneExpertOneBatch)
        logger.debug('LoadTimeOneExpert: %s',LoadTimeOneExpert)
        if self.if_usecpu: 
            CPUComputeTimeOneExpertOneBatch = remove_outliers_and_average(self.CPUComputeTimeOneExpertOneBatch)
            LoadTimeOneExpert = remove_outliers_and_average(self.ExpertCache.LoadTimeOneExpert) if len(self.CPUComputeTimeOneExpertOneBatch) >2 else CPUComputeTimeOneExpertOneBatch
            expertneedload,expertcomputeusecpu = CPU_load_management(uid_batch,CPUComputeTimeOneExpertOneBatch,LoadTimeOneExpert)
        else:
            expertneedload = list(uid_batch.keys())
            expertcomputeusecpu = []
        # expertneedload = expertnotincudalst[:load_num]
        # expertcomputeusecpu = expertnotincudalst[load_num:]
        logger.debug("expertcomputedirec %s",expertcomputeusecudaincache)
        logger.debug("expertneedload %s",expertneedload)
        logger.debug("expertcomputeusecpu %s",expertcomputeusecpu)
        logger.debug("self.CPUComputeTimeOneExpertOneBatch %s",self.CPUComputeTimeOneExpertOneBatch)
        #分配各专家需要的计算信息
        # global cache_ratio,times
        # cache_ratio+=(len(expertcomputeusecudaincache)/8)
        # times+=1
        # logger.debug("cacheratio %s",cache_ratio/times)
        e3 = time.time()
        expert_out_dict = dict()
        thread = threading.Thread(target=self.compute_expertsincache_in_thread, args=(
            expertcomputeusecudaincache,
            final_hidden_states,
            expert_token_dic,
            hdtype,
            expert_num_not_in_cuda_num,
            residual_cur,attn_weights_cur,present_key_value_cur,identity,attention_mask,position_ids,output_attentions,expert_out_dict,cache_position,position_embeddings,bsh
        ))
        thread.start()
        for uid in expertneedload:
            self.ExpertCache.add_to_queue(uid)
        e4 = time.time()
        # logger.info("time phase4,load time: %s",e4-e3)
        
        e5 = time.time()
        logger.debug("time phase5 %s",e5-e4)
        # torch.set_num_threads(1)
        for uid in expertcomputeusecpu: # zero
            expert  = self.ExpertCache.get_compute_expert(uid,True)
            # logger.debug("cpuexpert_tokens.device %s",expert.gate_proj.device)
            expert_tokens=expert_token_dic[uid][0]
            expert_batch = expert_tokens.size()[0]
            # logger.debug("expert.device %s",expert.device)
            expert_tokens = expert_tokens.to("cpu")
            startcom=time.time()
            expert_out = expert(expert_tokens)
            # logger.debug(prof.key_averages().table(sort_by="self_cuda_time_total %s", row_limit=10))
            endcom=time.time()
            logger.debug("num_thread: %s",torch.get_num_threads())
            global CPUtime
            if modeling_qwen.tokens>0:
                modeling_qwen.CPUtime += (endcom-startcom)
                # rint("compute one expert time in CPU: %s %s",uid,endcom-startcom) #when not use CPU ,its not triggered
            # print("expert batch: %s",expert_batch)
            expert_out=expert_out.to(self.config.device)
            
            weight= expert_token_dic[uid][1]
            top_x = expert_token_dic[uid][2]
            expert_out.mul_(weight)
            # with self.hiddenstatelock: 
            #     expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
            expert_out_dict[uid] = expert_out
            if len(self.CPUComputeTimeOneExpertOneBatch)<10:
                self.CPUComputeTimeOneExpertOneBatch.append( endcom-startcom)
            else:
                self.CPUComputeTimeOneExpertOneBatch.append(endcom-startcom)
                self.CPUComputeTimeOneExpertOneBatch = self.CPUComputeTimeOneExpertOneBatch[-10:]
        self.ExpertCache.wait_until_queue_empty()
        thread.join()
        if self.next_experts !=None:
            s= time.time()
            for experts_id in self.next_experts:
                uid = (self.layerid+1,experts_id)
                self.ExpertCache.add_to_queue(uid)
                logger.debug("prefetch %s",uid)
            e=time.time()
            logger.debug("loadtimeforexp %s",e-s)
        e6 = time.time()
        logger.debug("time phase6 %s",e6-e5)
        for uid in expertneedload:
            
            expert = self.ExpertCache.get_compute_expert(uid)
            startcom=time.time()
            expert_tokens=expert_token_dic[uid][0]
            expert_out = expert(expert_tokens)
            weight= expert_token_dic[uid][1]
            expert_out.mul_(weight)
            expert_out_dict[uid] = expert_out
            endcom=time.time()
            global compute_time_1
            compute_time_1 += endcom-startcom
            # logger.info("compute time in cuda2: %s",endcom-startcom)
            compute_time_1 += endcom-startcom

        e7 = time.time()
        logger.debug("time phase7 %s",e7-e6)
        for uid,_ in expert_token_dic.items():
            top_x = expert_token_dic[uid][2]
            expert_out = expert_out_dict[uid]
            final_hidden_states.index_add_(0, top_x, expert_out.to(hdtype))
        e = time.time()
        logger.debug("time phase8 %s",e-e7)
        logger.debug("a token time:  %s", e-s)
        logger.debug("--------------")
    
    @torch.no_grad()
    def get_next_top_expert(self,residual_cur,attn_weights_cur,present_key_value_cur,hidden_states,raw_hidden,attention_mask,position_ids,output_attentions,cache_position,position_embeddings,bsh):
        shared_expert_output = self.shared_expert(raw_hidden)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(raw_hidden)) * shared_expert_output
        hidden_states=hidden_states+shared_expert_output  
        hidden_states = hidden_states.reshape(bsh[0], bsh[1], bsh[2])
        hidden_states = residual_cur+hidden_states # formal
        # hidden_states = residual_cur  # sit1
        # hidden_states = residual_cur + shared_expert_output # sit2
        # hidden_states = hidden_states - shared_expert_output #sit4
        outputs = (hidden_states,)
        outputs += (attn_weights_cur,)
        outputs += (present_key_value_cur,)
        next_hidden_states = outputs[0]
        next_residual = next_hidden_states
        next_hidden_states = self.next_input_layernorm(next_hidden_states)
        # next_hidden_states, _, _ = self.next_attention(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_value=present_key_value_cur,
        #     output_attentions=output_attentions,
        #     use_cache=True,
        #     cache_position=cache_position,
        #     position_embeddings=position_embeddings,
        #     if_update = False
        # )
        next_hidden_states =next_hidden_states+next_residual
        next_hidden_states = self.next_post_attention_layernorm(next_hidden_states)
        bsz, seq_len, h = next_hidden_states.shape
        ### compute gating score
        next_hidden_states = next_hidden_states.view(-1, h)
        # logger.debug(hidden_states)
        # logger.debug(self.weight)
        logits = self.next_gate_weight(next_hidden_states)
        scores = F.softmax(logits, dim=1, dtype=torch.float)
        ### select top-k experts
        # s, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=True)
        topexperts,_ = replaceset_between_tokens(scores.tolist(),self.replaceScoreRatio,self.top_k)
        # res = [item for sublist in topexperts for item in sublist]
        return topexperts
        # only when test prefetch
        # next_expert_tops,_ = replaceset_between_tokens(scores.tolist(),self.replaceScoreRatio,self.top_k)
        # logger.info("next experts tops %s, layer %s",next_expert_tops, self.layerid+1)