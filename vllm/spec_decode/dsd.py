from vllm.sequence import ExecuteModelRequest
from typing import Dict, Optional, Tuple
from vllm.worker.model_runner import _get_graph_batch_size
from vllm.logger import init_logger

logger = init_logger(__name__)


class DSD:

    def __init__(self,
                 fixed_acceptance_rate: Optional[float] = None,
                 draft_times_map: Optional[Dict[int, Dict]] = None,
                 target_times_map: Optional[Dict[int, Dict]] = None):
        # Global token acceptance rate for now
        self.token_acceptance_rate = fixed_acceptance_rate
        if self.token_acceptance_rate is not None:
            logger.info("[DSD] Using initial token acceptance rate %f",
                        self.token_acceptance_rate)

        self.compute_coefficient = 0
        self.load_kv_coefficient = 0
        self.load_param_coefficient = 0

        self.draft_times_map = draft_times_map
        self.target_times_map = target_times_map

        self.is_ngram = False
        print("=" * 40)
        print(f"Draft times map: {self.draft_times_map}")
        print(f"Target times map: {self.target_times_map}")
        print("=" * 40)

    def _predict_goodput(self, batch: ExecuteModelRequest, k: int) -> float:
        accepted_len = self._get_accepted_len(batch, k)
        batch_time = self._get_batch_time(batch, k)
        return accepted_len / batch_time

    def _get_accepted_len(self, batch: ExecuteModelRequest, k: int) -> float:
        batch_size = len(batch.seq_group_metadata_list)
        assert self.token_acceptance_rate is not None
        acc_len = float((1 - self.token_acceptance_rate**(k + 1)) /
                        (1 - self.token_acceptance_rate)) * batch_size
        # print(f"k: {k}, Accepted len: {acc_len}")
        return acc_len

    def _get_batched_kv_token(self, batch: ExecuteModelRequest,
                              k: int) -> Tuple[int, int]:
        num_batched_token = 0
        num_kv_token = 0
        for seq_group_metadata in batch.seq_group_metadata_list:
            assert len(seq_group_metadata.seq_data) == 1
            seq_id = seq_group_metadata.seq_data.keys()[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            num_batched_token += k + 1
            num_kv_token += seq_data.get_len()
        return num_batched_token, num_kv_token

    def _get_bucket_seq_len(self, times_map: Dict[int, float],
                            seq_len: int) -> int:
        all_seq_lens = list(times_map.keys())
        all_seq_lens.sort()
        for i in range(len(all_seq_lens) - 1):
            if all_seq_lens[i] <= seq_len and seq_len < all_seq_lens[i + 1]:
                return all_seq_lens[i]
        raise AssertionError(f"Seq len {seq_len} not found in times map")

    def _get_batch_avg_seq_len(self, batch: ExecuteModelRequest) -> int:
        total_seq_len = 0
        for seq_group_metadata in batch.seq_group_metadata_list:
            assert len(seq_group_metadata.seq_data) == 1
            seq_id = list(seq_group_metadata.seq_data.keys())[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            total_seq_len += seq_data.get_len()
        return total_seq_len // len(batch.seq_group_metadata_list)

    def _get_batch_time(self, batch: ExecuteModelRequest, k: int) -> float:
        assert self.draft_times_map is not None
        assert self.target_times_map is not None
        batch_size = len(batch.seq_group_metadata_list)
        draft_graph_batch_size = _get_graph_batch_size(batch_size)
        avg_seq_len = self._get_batch_avg_seq_len(batch)
        seq_len = self._get_bucket_seq_len(self.draft_times_map, avg_seq_len)

        single_draft_time = self.draft_times_map[seq_len][
            draft_graph_batch_size]
        draft_time = single_draft_time * k

        num_batched_token = (k + 1) * batch_size
        target_graph_batch_size = _get_graph_batch_size(num_batched_token)
        target_time = self.target_times_map[seq_len][target_graph_batch_size]
        # print(f"Draft time: {draft_time}, Target time: {target_time}")
        return draft_time + target_time

    def get_propose_len(self, batch: ExecuteModelRequest) -> int:
        if self.is_ngram:
            return 10  # Hardcode a very large propose lengh for ngram

        max_proposal_len = batch.num_lookahead_slots
        max_goodput = -1.0
        best_proposal_len = -1
        for i in range(1, max_proposal_len + 1):
            cur_goodput: float = self._predict_goodput(batch, i)
            # print(f"Goodput for proposal len {i}: {cur_goodput}")
            if cur_goodput > max_goodput:
                max_goodput = cur_goodput
                best_proposal_len = i
        logger.info(f"==Best proposal len: {best_proposal_len}")
        logger.info(self.draft_times_map is None)
        return best_proposal_len

    def get_verify_len(self, batch: ExecuteModelRequest,
                       proposal_len: int) -> int:
        if not self.is_ngram:
            return proposal_len

    def set_token_acceptance_rate(self, token_acceptance_rate: float):
        self.token_acceptance_rate = token_acceptance_rate
