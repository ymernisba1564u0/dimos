
import dimos.protocol.service.lcmservice as _lcm_mod
import dimos.protocol.rpc.pubsubrpc as _rpc_mod

# 1. Increase LCM polling timeout: 50ms -> 200ms
#    Reduces context switches from ~15k/sec to ~3.75k/sec
_lcm_mod._LCM_LOOP_TIMEOUT = 200

# 2. Reduce RPC thread pool: 50 -> 4 workers per module
#    During replay, RPC calls are minimal
_rpc_mod.PubSubRPCBase._call_thread_pool_max_workers = 4
