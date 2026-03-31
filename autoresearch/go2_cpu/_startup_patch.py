
import dimos.protocol.service.lcmservice as _lcm_mod
import dimos.protocol.rpc.pubsubrpc as _rpc_mod

# 1. Increase LCM polling timeout: 50ms -> 200ms
#    750 threads × 20 polls/sec = 15k context switches/sec
#    With 200ms: 750 × 5 polls/sec = 3.75k context switches/sec
_lcm_mod._LCM_LOOP_TIMEOUT = 200

# 2. Reduce RPC thread pool: 50 -> 4 workers per module
#    During replay, RPC calls are minimal — no real robot commands
_rpc_mod.PubSubRPCBase._call_thread_pool_max_workers = 4
