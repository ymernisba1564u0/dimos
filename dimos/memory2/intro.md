# Memory Intro

## Quick start

```python session=memory ansi=false no-result
from dimos.memory2.impl.sqlite import SqliteStore

store = SqliteStore(path="/tmp/memory_readme.db")
session = store.session()
```


```python session=memory ansi=false
logs = session.stream("logs", str)
print(logs)
```

<!--Result:-->
```
Stream("logs")
```

Append observations and query them:

```python session=memory ansi=false
logs.append("Motor started", ts=1.0, tags={"level": "info"})
logs.append("Joint 3 fault", ts=2.0, tags={"level": "error"})
logs.append("Motor stopped", ts=3.0, tags={"level": "info"})

print(logs.summary())
```

<!--Result:-->
```
Stream("logs"): 3 items, 1970-01-01 00:00:01 — 1970-01-01 00:00:03 (2.0s)
```

### Filters

Queries are lazy — chaining filters builds a pipeline without fetching:

```python session=memory ansi=false
print(logs.at(1.0).before(5.0).tags(level="error"))
```

<!--Result:-->
```
Stream("logs") | AtFilter(t=1.0, tolerance=1.0) | BeforeFilter(t=5.0) | TagsFilter(tags={'level': 'error'})
```

Available filters: `.after(t)`, `.before(t)`, `.at(t)`, `.near(pose, radius)`, `.tags(**kv)`, `.filter(predicate)`, `.search(embedding, k)`, `.order_by(field)`, `.limit(k)`, `.offset(n)`.

### Terminals

Terminals materialize or consume the stream:

```python session=memory ansi=false
print(logs.before(5.0).tags(level="error").fetch())
```

<!--Result:-->
```
[Observation(id=2, ts=2.0, pose=None, tags={'level': 'error'})]
```

Available terminals: `.fetch()`, `.first()`, `.last()`, `.count()`, `.exists()`, `.summary()`, `.get_time_range()`, `.drain()`, `.save(target)`.

### Transforms

`.map(fn)` transforms each observation, returning a new stream:

```python session=memory ansi=false
print(logs.map(lambda obs: obs.data.upper()).first())
```

<!--Result:-->
```
MOTOR STARTED
```

### Live queries

Live queries backfill existing matches, then emit new ones as they arrive:

```python session=memory ansi=false
import time

def emit_some_logs():
    last_ts = logs.last().ts
    logs.append("Heartbeat ok", ts=last_ts + 1, pose=(3.0, 1.5, 0.0), tags={"level": "info"})
    time.sleep(0.1)
    logs.append("Sensor fault", ts=last_ts + 2, pose=(4.1, 2.0, 0.0), tags={"level": "error"})
    time.sleep(0.1)
    logs.append("Battery charge 30%", ts=last_ts + 3, pose=(5.3, 2.5, 0.0), tags={"level": "info"})
    time.sleep(0.1)
    logs.append("Overtemp", ts=last_ts + 4, pose=(6.0, 3.0, 0.0), tags={"level": "error"})
    time.sleep(0.1)


with logs.tags(level="error").live() as errors:
    sub = errors.subscribe(lambda obs: print(f"{obs.ts} - {obs.data}"))
    emit_some_logs()
    sub.dispose()

```

<!--Result:-->
```
2.0 - Joint 3 fault
5.0 - Sensor fault
7.0 - Overtemp
```

## Spatial + live

Filters compose freely. Here `.near()` + `.live()` + `.map()` watches for logs near a physical location — backfilling past matches and tailing new ones:

```python session=memory ansi=false

with logs.near((5.0, 2.0), radius=2.0).live().map(lambda obs: f"log entry around our point of interest - {obs.data}") as logs_near:
    # subscription is also contextmanager
    with logs_near.subscribe(print):
        emit_some_logs()
```

<!--Result:-->
```
log entry around our point of interest - Sensor fault
log entry around our point of interest - Battery charge 30%
log entry around our point of interest - Overtemp
log entry around our point of interest - Sensor fault
log entry around our point of interest - Battery charge 30%
log entry around our point of interest - Overtemp
```
