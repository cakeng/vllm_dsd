from vllm.sequence import ExecuteModelRequest
from typing import Dict
from vllm.worker.model_runner import _get_graph_batch_size

class DSD:
    def __init__(self,
                 draft_times_map: Dict[int, float],
                 target_times_map: Dict[int, float]):
        # Global token acceptance rate for now
        self.token_acceptance_rate = 0.7
        self.compute_coefficient = 0
        self.load_kv_coefficient = 0
        self.load_param_coefficient = 0
        
        self.draft_times_map = draft_times_map
        self.target_times_map = target_times_map
        
    
    def _predict_goodput(self, 
                         batch: ExecuteModelRequest,
                         k: int) -> float:
        accepted_len = self._get_accepted_len(batch, k)
        batch_time = self._get_batch_time(batch, k)
        return accepted_len / batch_time
    
    def _get_accepted_len(self, 
                          batch: ExecuteModelRequest,
                          k: int) -> float:
        batch_size = len(batch.seq_group_metadata_list)
        acc_len = float((1 - self.token_acceptance_rate ** (k + 1)) / (1 - self.token_acceptance_rate)) * batch_size
        print(f"k: {k}, Accepted len: {acc_len}")
        return acc_len
    
    def _get_batched_kv_token(self, 
                              batch: ExecuteModelRequest,
                              k: int) -> int:
        num_batched_token = 0
        num_kv_token = 0
        for seq_group_metadata in batch.seq_group_metadata_list:
            assert len(seq_group_metadata.seq_data) == 1
            seq_id = seq_group_metadata.seq_data.keys()[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            num_batched_token += k + 1
            num_kv_token += seq_data.get_len()
        return num_batched_token, num_kv_token
        
    def _get_batch_time(self, 
                        batch: ExecuteModelRequest,
                        k: int) -> float:
        estimate_method = "profile"
        
        if estimate_method == "linear":
            raise NotImplementedError("Linear estimate method is not fully implemented")
            draft_time = TODO
            num_batched_token, num_kv_token = self._get_batched_kv_token(batch, k)
            target_time = self.compute_coefficient * num_batched_token + \
                self.load_kv_coefficient * num_kv_token + \
                    self.load_param_coefficient
            return draft_time + target_time
        elif estimate_method == "profile":
            batch_size = len(batch.seq_group_metadata_list)
            draft_graph_batch_size = _get_graph_batch_size(batch_size)
            single_draft_time = self.draft_times_map[draft_graph_batch_size]
            draft_time = 0.05 * single_draft_time * k + 0.95 * single_draft_time
            
            num_batched_token = (k + 1) * batch_size
            target_graph_batch_size = _get_graph_batch_size(num_batched_token)
            target_time = self.target_times_map[target_graph_batch_size]
            print(f"Draft time: {draft_time}, Target time: {target_time}")
            return draft_time + target_time
        else:
            raise ValueError(f"Invalid estimate method {estimate_method}")
            
            
    def get_propose_len(self, batch: ExecuteModelRequest) -> int:
        max_proposal_len = batch.num_lookahead_slots
        max_goodput = -1
        best_proposal_len = -1
        for i in range(1, max_proposal_len + 1):
            cur_goodput: float = self._predict_goodput(batch, i)
            print(f"Goodput for proposal len {i}: {cur_goodput}")
            if cur_goodput > max_goodput:
                max_goodput = cur_goodput
                best_proposal_len = i
        print(f"===============Best proposal len: {best_proposal_len}")
        return best_proposal_len
                
    def get_verify_len(self, 
                       batch: ExecuteModelRequest,
                       proposal_len: int) -> int:
        return proposal_len
    
    
    
    
    