# REPL

The REPL gives you a live Python shell connected to a running DimOS instance. You can inspect module state, call methods, and debug without restarting.

## Quick Start

Start DimOS (the REPL server is enabled by default):

```bash
dimos run unitree-go2
```

In another terminal, connect:

```bash
dimos repl
```

You get an interactive Python session with these pre-loaded objects:

| Name | Description |
|------|-------------|
| `coordinator` | The `ModuleCoordinator` instance |
| `modules()` | List deployed module class names |
| `get(name)` | Get a live module instance by class name |

## Examples

```python
# List all deployed modules
>>> modules()
['GO2Connection', 'RerunBridge', 'McpServer', ...]

# Get a module instance and call methods on it
>>> wfe = get('WavefrontFrontierExplorer')
>>> wfe.begin_exploration()
"Started exploring."

# Access the coordinator directly
>>> coordinator.list_modules()
['GO2Connection', 'RerunBridge', ...]
```

## How It Works

The REPL uses [RPyC](https://rpyc.readthedocs.io/) for transparent remote object access. When `dimos run` starts, it launches:

1. A **coordinator RPyC server** on the main process (default port `18861`). This is the entry point for `dimos repl`.
2. A **worker RPyC server** inside each worker process (auto-assigned ports). When you call `get("ModuleName")`, the REPL connects directly to that module's worker process.

This means `get()` returns a live proxy to the actual module object in its worker process. Attribute access, method calls, and return values are transparently proxied over the network.

## CLI Reference

### `dimos run` Options

| Option | Default | Description |
|--------|---------|-------------|
| `--repl` / `--no-repl` | `--repl` (enabled) | Enable or disable the RPyC REPL server |
| `--repl-port` | `18861` | Port for the coordinator REPL server |

```bash
# Disable the REPL server
dimos run unitree-go2 --no-repl

# Use a custom port
dimos run unitree-go2 --repl-port 19000
```

### `dimos repl`

Connect to a running DimOS instance.

```bash
dimos repl [--host HOST] [--port PORT]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host`, `-H` | `localhost` | Host to connect to |
| `--port`, `-p` | auto-detected | REPL server port (reads from run registry if omitted) |

The port is auto-detected from the run registry. You only need `--port` if you used a custom `--repl-port` and there is no active run entry (e.g., non-daemon foreground run that was killed).

If IPython is installed, the REPL uses it automatically for tab completion, syntax highlighting, and history. Otherwise it falls back to the standard Python REPL.
