import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
import random
from expertcachewithscorefullcpu import ExpertCache,cache_router,remove_outliers_and_average,CPU_load_management,replaceset_between_tokens

from typing import Tuple,List,OrderedDict
import time
import numpy as np
import threading
ExpertUID = Tuple[int,int]
cache_ratio = 0
times =0
top_loads =0
low_loads = 0
all_loads = 0
tops =0
lows=0
scores_all =[0,0,0,0,0,0,0,0,0,0]
scoretime=0
prefetch_experts = []
prefetch_hit_sum = 0
load_sum = 0
generate_length = 31
prefetch_number = 0
least_batch = 10000
import logging
logger = logging.getLogger(__name__)




class DeepseekMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False,dtype=torch.bfloat16, device=config.device)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False,dtype=torch.bfloat16, device=config.device)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False,dtype=torch.bfloat16, device=config.device)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj



class DeepseekMoEwithCache(nn.Module):
    def __init__(
        self,config,expertcache: ExpertCache, layerid,gate:nn.Linear,shared_experts,next_attention,next_gate_weight,next_input_layernorm,next_post_attention_layernorm
    ):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts if config.n_shared_experts is not None else None

        self.layerid = layerid
        self.router = gate
        self.ExpertCache = expertcache
        self.shared_experts = shared_experts
        
        self.next_attention = next_attention
        self.next_gate_weight = next_gate_weight
        self.next_input_layernorm = next_input_layernorm
        self.next_post_attention_layernorm = next_post_attention_layernorm
        self.next_experts = None
        
        self.if_usecpu = config.if_usecpu
        self.if_prefetch = config.if_prefetch
        self.replaceScoreRatio = config.replaceScoreRatio
        self.CPUComputeTimeOneExpertOneBatch=[0.05]

    def forward(self, hidden_states,residual_cur,attn_weights_cur,present_key_value_cur,attention_mask,position_ids,output_attentions):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # logger.info("layer i hidden_states:%s", hidden_states)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, hidden_dim)
        #和另外两个不同，gate是nn.parameters
        router_logits  = F.linear(hidden_states, self.router, None)
        router_logits = router_logits.view(-1,router_logits.size()[-1])
        Scores = F.softmax(router_logits, dim=1, dtype=torch.float)#seq*64
        if self.ExpertCache.cache_window!=None:
            for score_l in Scores.tolist():
                self.ExpertCache.update_scores(self.layerid, score_l)
        
        routing_weights, selected_experts = torch.topk(Scores, self.top_k, dim=-1, sorted=True)#197*6
        
        
        s=Scores.tolist()
        # sorted_s = sorted(s[0], reverse=True)#delete
        # logger.info('layer: %s, Scores: %s',self.layerid, sorted_s[:12])
        sort_index = [sorted(range(len(input_list)),key=lambda i:input_list[i],reverse=True) for input_list in s]
        sort_scores = [[s[j][i] for i in sort_index[j]] for j in range(len(s))]
        if len(sort_scores)==1:
            logger.debug("true score %s",sort_scores[0][0:12])
            logger.debug("true expert %s",sort_index[0][0:12])
        global scores_all,scoretime
        for j in range(len(sort_scores)):
            scoretime+=1
            for i in range(10):
                scores_all[i]+=sort_scores[j][i]
        logger.debug("scoretime %s",scoretime)
        for i in scores_all:
            logger.debug(i/scoretime)
        
        replaceset = []

        if self.replaceScoreRatio!=None:
            replaceset,allset = replaceset_between_tokens(Scores.tolist(),self.replaceScoreRatio,self.top_k)
            
            global tops ,lows

                # logger.info('layer i tops: %s, layer %s',replaceset,self.layerid)
            # logger.debug("top ratio %s",tops/(lows+tops))
            if lows+tops > 0:
                # logger.info("low ratio %s",lows/(lows+tops))
                pass

            cacherouter_experts,_ = cache_router(Scores.tolist(),self.ExpertCache,self.replaceScoreRatio,self.top_k,replaceset,self.layerid)
            selected_experts = torch.tensor(cacherouter_experts,device=routing_weights.device)

                 
            routing_weights = Scores[torch.arange(Scores.size(0)).unsqueeze(1),selected_experts]
        for topl in selected_experts.tolist():
            for expert_id in topl:
                uid = (self.layerid,expert_id)
                self.ExpertCache.ready_compute(uid)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts)#1*6
        expert_mask = expert_mask.permute(2, 1, 0)#64*6*1

        routing_weights /= (routing_weights.sum(dim=-1, keepdim=True) + 1e-06)

        routing_weights = routing_weights.to(input_dtype)
        hidden_states = hidden_states.to(input_dtype)
        
        # if hidden_states.size()[0] == 1: 
        expert_token_dic=dict()
        topidxlst = []
        for l in selected_experts.tolist():
            for uid in l:
                topidxlst.append(uid)
        experttokens3 = 0 
        for expert_idx in range(self.num_experts):
            if expert_idx not in topidxlst:
                continue
            uid = (self.layerid,expert_idx)
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            expert_token_dic[uid] = [current_state,routing_weights[top_x, idx, None],top_x]
            # print("uid and expert tokens",uid,current_state.size()[0])
            if current_state.size()[0]>3:
                experttokens3+=1
        print("experttokens3",experttokens3/64)
        #根据expert_token_dic和self.expertcache运算得到final_hidden_states
        self.moe_infer_for_one_batch(replaceset,final_hidden_states,expert_token_dic,hidden_states.dtype,residual_cur,attn_weights_cur,present_key_value_cur,hidden_states,attention_mask,position_ids,output_attentions,(batch_size,sequence_length,hidden_dim))
        # else:
        #     expert_token_dic=dict()
        #     topidxlst = selected_experts.tolist()
        #     topidxlst = [item for sublist in topidxlst for item in sublist]
        #     for expert_idx in range(self.num_experts):
        #         if expert_idx not in topidxlst:
        #             continue
        #         uid = (self.layerid,expert_idx)
        #         idx, top_x = torch.where(expert_mask[expert_idx])
        #         current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        #         expert_token_dic[uid] = [current_state,routing_weights[top_x, idx, None],top_x]
        #     self.moe_infer(final_hidden_states,expert_token_dic,hidden_states.dtype)
        for l in selected_experts.tolist():
            for i in l:
                uid = (self.layerid,i)
                self.ExpertCache.end_compute(uid)

            # logger.info("predict layer i+1 next_experts %s",self.next_experts)# only 6

            # replaceset,allset = replaceset_between_tokens(Scores.tolist(),self.replaceScoreRatio,self.top_k)
            # ,_ = cache_router(Scores.tolist(),self.ExpertCache,self.replaceScoreRatio,self.top_k,replaceset,self.layerid)
            

            prefetch_experts = self.next_experts
            # logger.info("prefetch accuracy: %s %s %s", )
        global prefetch_number
        if self.next_experts !=None:
            s= time.time()
            prefetch_number += 1
            for experts_id in self.next_experts:
                uid = (self.layerid+1,experts_id)
                self.ExpertCache.add_to_queue(uid)

                
            e=time.time()
            logger.debug("loadtimeforexp %s",e-s)
        logger.debug("begincomputeshared")

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if self.num_shared_experts is not None:
            hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
            shared_hidden = self.shared_experts(hidden_states)
            final_hidden_states = final_hidden_states + shared_hidden

        return final_hidden_states, router_logits
    
    def compute_expertsincache_in_thread(self,expertcomputeusecudaincache,x, expert_token_dic,hdtype,expert_num_not_in_cuda_num,residual_cur,attn_weights_cur,present_key_value_cur,identity,attention_mask,position_ids,output_attentions,expert_out_dict,bsh):
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
            expert_batch = expert_tokens.size()[0]
            weight = expert_token_dic[uid][1]
            top_x = expert_token_dic[uid][2]
            startcom = time.time()
            expert_out = expert(expert_tokens)
            endcom = time.time()
            logger.info("experts compute cuda in new_thread: %s %s", uid, endcom - startcom)
            print("compute one expert time in GPU: %s %s",uid,endcom-startcom) #when not use CPU ,its not triggered
            print("expert batch: %s",expert_batch)
            # logger.debug("computetime cuda in new_thread %s %s", endcom - startcom, uid)
            expert_out.mul_(weight)
            expert_out_dict[uid] = expert_out
            hidden_states.index_add_(0, top_x, expert_out.to(hdtype))
            s3 = time.time()
            logger.debug("last %s",s3-endcom)
        if self.layerid <27 and expert_num_not_in_cuda_num>0 and self.if_prefetch :
            s=time.time()
            self.next_experts = self.get_next_top_expert(residual_cur,attn_weights_cur,present_key_value_cur,hidden_states,identity,attention_mask,position_ids,output_attentions,bsh)
            e =time.time()
            logger.debug("predicttime %s",e-s)
            logger.debug("next_experts %s",self.next_experts)
            # 计算 prefetch 准确率
            
    @torch.no_grad()
    def moe_infer(self, final_hidden_states,expert_token_dic,hdtype):
        for uid in expert_token_dic.keys():
            self.ExpertCache.add_to_queue(uid)
        self.ExpertCache.wait_until_queue_empty()
        torch.cuda.synchronize()

        logger.debug("---------")
        expert_out_dict = dict()
        
        for uid in expert_token_dic.keys():

            expert = self.ExpertCache.get_compute_expert(uid)
            expert_tokens=expert_token_dic[uid][0]
            logger.info('expert: %s', expert)
            expert_out = expert(expert_tokens)

            weight= expert_token_dic[uid][1]
            expert_out.mul_(weight)
            expert_out_dict[uid] = expert_out

        for uid,_ in expert_token_dic.items():
            top_x = expert_token_dic[uid][2]
            expert_out = expert_out_dict[uid]
            final_hidden_states.index_add_(0, top_x, expert_out.to(hdtype))

    @torch.no_grad()
    def moe_infer_for_one_batch(self, replaceset,final_hidden_states, expert_token_dic,hdtype,residual_cur,attn_weights_cur,present_key_value_cur,identity,attention_mask,position_ids,output_attentions,bsh):
        logger.debug("--------------")
        s= time.time()
        #分配哪些需要加载，哪些需要CPU计算,哪些专家可以直接cuda计算
        expertnotincudalst = []
        expertcomputeusecudaincache = []
        expert_num_not_in_cuda_num=0
        # logger.debug("beforequeue",self.ExpertCache.load_queue)
        self.ExpertCache.clear_queue()
        self.ExpertCache.load_stream.synchronize()
        uid_batch = dict()
        
        newreplace=[]
        for i in replaceset:
            newreplace.append((self.layerid,i))
        logger.debug("topexperts %s",newreplace)
        # logger.debug("topratio",len(newreplace)/18)
        for uid in expert_token_dic.keys():
            if not self.ExpertCache.query_expert(uid): #cpu
                expert_num_not_in_cuda_num+=1
                expertnotincudalst.append(uid)
                uid_batch[uid] = expert_token_dic[uid][0].size()[0]#load experts list
                global top_loads,low_loads,all_loads


                
                # logger.info("lowloadratio",low_loads/all_loads)
                # logger.info("top CPU load ratio",top_loads/all_loads)
                
            else:
                expertcomputeusecudaincache.append(uid)
        global tops,lows
        if tops != 0:
            logger.debug("load_top_ratio %s",top_loads/tops)
        if lows != 0:
            logger.debug("load_low_ratio %s",low_loads/lows)
        if self.if_usecpu: 
            CPUComputeTimeOneExpertOneBatch = remove_outliers_and_average(self.CPUComputeTimeOneExpertOneBatch)
            LoadTimeOneExpert = remove_outliers_and_average(self.ExpertCache.LoadTimeOneExpert) if len(self.CPUComputeTimeOneExpertOneBatch) >2 else CPUComputeTimeOneExpertOneBatch
            expertneedload,expertcomputeusecpu = CPU_load_management(uid_batch,CPUComputeTimeOneExpertOneBatch,LoadTimeOneExpert)
        else:
            expertneedload = list(uid_batch.keys())
            expertcomputeusecpu = []
        # expertneedload = expertnotincudalst[:expert_num_not_in_cuda_num//2]
        # expertcomputeusecpu = expertnotincudalst[expert_num_not_in_cuda_num//2:]
        logger.debug("expertcomputedirec %s",expertcomputeusecudaincache)
        logger.debug("expertneedload %s",expertneedload)
        logger.debug("expertcomputeusecpu %s",expertcomputeusecpu)
        #logger.debug("self.CPUComputeTimeOneExpertOneBatch",self.CPUComputeTimeOneExpertOneBatch)
        #分配各专家需要的计算信息
        global cache_ratio,times
        cache_ratio+=(len(expertcomputeusecudaincache)/6)
        times+=1
        logger.debug("cacheratio %s",cache_ratio/times)
        e3 = time.time()
        expert_out_dict = dict()
        thread = threading.Thread(target=self.compute_expertsincache_in_thread, args=(
            expertcomputeusecudaincache,
            final_hidden_states,
            expert_token_dic,
            hdtype,
            expert_num_not_in_cuda_num,
            residual_cur,attn_weights_cur,present_key_value_cur,identity,attention_mask,position_ids,output_attentions,expert_out_dict,bsh
        ))
        thread.start()
        for uid in expertneedload:
            self.ExpertCache.add_to_queue(uid)
        e4 = time.time()
        logger.debug("time phase4 %s",e4-e3)
        e5 = time.time()
        logger.debug("time phase5 %s",e5-e4)
        for uid in expertcomputeusecpu:
            expert  = self.ExpertCache.get_compute_expert(uid,True)
            # logger.debug("cpuexpert_tokens.device",expert.gate_proj.device)
            expert_tokens=expert_token_dic[uid][0]
            # logger.debug("expert.device",expert.device)
            # logger.debug("expert_outtyperaw %s",expert_tokens.dtype)
            # with torch.cpu.amp.autocast(enabled=False):
            expert_tokens = expert_tokens.to("cpu")
            expert_batch = expert_tokens.size()[0]
            # logger.debug("expert_outtypeafter %s",expert_tokens.dtype)
            startcom=time.time()
            expert_out = expert(expert_tokens)
            # logger.debug("expert_outtype %s",expert_out.dtype)
            # logger.debug(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
            endcom=time.time()
            logger.debug("num_thread: %s",torch.get_num_threads())
            logger.info("compute one expert time in CPU: %s %s",uid,endcom-startcom)
            print("compute one expert time in CPU: %s %s",uid,endcom-startcom) #when not use CPU ,its not triggered
            print("expert batch: %s",expert_batch)
            expert_out=expert_out.to(self.config.device)
            # logger.debug("computetimeCPU %s",endcom-startcom,uid)
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
        self.ExpertCache.load_stream.synchronize()
        e6 = time.time()
        logger.debug("time phase6: %s",e6-e5)
        for uid in expertneedload:
            expert = self.ExpertCache.get_compute_expert(uid)
            expert_tokens=expert_token_dic[uid][0]
            startcom=time.time()
            expert_out = expert(expert_tokens)
            endcom=time.time()
            logger.debug("computetimecuda: time=%.4f | uid=%s",endcom-startcom,uid)
            weight= expert_token_dic[uid][1]
            expert_out.mul_(weight)
            # logger.debug(expert_out)
            
            expert_out_dict[uid] = expert_out
            
            # logger.debug("testcpu")
            # expert  = self.ExpertCache.get_compute_expert(uid,True)
            # expert_tokens=expert_token_dic[uid][0]
            # expert_tokens = expert_tokens.to("cpu")
            # expert_out = expert(expert_tokens)
            # logger.debug(expert_out)
            
        thread.join()
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
    def get_next_top_expert(self,residual_cur,attn_weights_cur,present_key_value_cur,hidden_states,raw_hidden,attention_mask,position_ids,output_attentions,bsh):
        #         final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        # if self.num_shared_experts is not None:
        #     hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        #     shared_hidden = self.shared_experts(hidden_states)
        #     final_hidden_states = final_hidden_states + shared_hidden
        hidden_states = hidden_states.reshape(bsh[0], bsh[1], bsh[2])
        raw_hidden = raw_hidden.reshape(bsh[0], bsh[1], bsh[2])
        shared_expert_output = self.shared_experts(raw_hidden)
        hidden_states=hidden_states+shared_expert_output 
        # hidden_states = residual_cur #situation1
        # hidden_states = residual_cur + shared_expert_output   #situation2
        hidden_states = residual_cur+hidden_states # sit 3 formal
        outputs = (hidden_states,)
        outputs += (attn_weights_cur,)
        outputs += (present_key_value_cur,)
        next_hidden_states = outputs[0]
        next_residual = next_hidden_states
        next_hidden_states = self.next_input_layernorm(next_hidden_states)
        next_hidden_states, _, _ = self.next_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=present_key_value_cur,
            output_attentions=output_attentions,
            use_cache=True,
            if_update = False
        )
        next_hidden_states =next_hidden_states+next_residual
        next_hidden_states = self.next_post_attention_layernorm(next_hidden_states)
        bsz, seq_len, h = next_hidden_states.shape
        ### compute gating score
        next_hidden_states = next_hidden_states.view(-1, h)
        # logger.debug(hidden_states)
        # logger.debug(self.weight)
        logger.debug("layer i+1 next_hidden_states %s", next_hidden_states)
        # logits  = F.linear(hidden_states,  self.next_gate_weight, None)
        logits  = F.linear(next_hidden_states,  self.next_gate_weight, None)
        logits = logits.view(-1,logits.size()[-1])
        scores = logits.softmax(dim=-1)
        ### select top-k experts
        s, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=True)
        logger.debug("topk_idx %s",topk_idx.size())

        topexperts = replaceset_between_tokens(s,topk_idx)
        if len(s)==1:
            assert(len(s[0]) == 6)
            # logger.info("next experts predict score %s, layer %s",s[0][0:12], self.layerid)
            # next_expert_tops,_ = replaceset_between_tokens(scores.tolist(),self.replaceScoreRatio,self.top_k)
            # logger.info("next experts tops %s, layer %s",next_expert_tops, self.layerid+1)
        # if modeling_deepseek.tokens > 0:
            # logger.info('topk_idx: %s', topk_idx)
        return topexperts