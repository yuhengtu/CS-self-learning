# Server Config Semantics

Written with genai assistance.

This project uses an nginx-like syntax, parsed into the existing `NginxConfig`
AST and then interpreted by our own semantics layer. The legacy parser already
tokenizes statements/blocks; the documentation below captures how we expect
configuration writers to describe handlers/routes so the dispatcher can build
its routing table.

## Syntax Overview

```nginx
server {
  listen 8080;

  location /echo {
    handler echo;
  }

  location /static {
    handler static;
    root ./www;
  }

  location /api {
    handler crud;
    data_path ./crud_data;
  }
}

### Top-level directives

| Directive | Required | Description |
| - | - | - |
| `server { ... }` | Yes | Top-level block defining a single server instance. Currently, only one `server` block is supported per config file. |
| `listen <number>;` | Yes | TCP port the server listens on. Must appear inside the `server` block. |
| `location /path { ... }` | Yes (≥1) | Declares a routed handler. Each `location` block defines the path it serves and contains a handler definition plus its options. Multiple `location` blocks are allowed. |

### Handler block fields

| Field | Required | Description |
| - | - | - |
| `handler <type>;` | Yes | Handler implementation identifier. Today we support `echo` (built-in), `static` (file serving), and crud (Create, Read, Update, and Delete). |
| Additional key/value pairs | optional | Passed to the handler factory. For example `root` for static files, 'data_path'for crud files or future custom options. Keys/values are single tokens; quoted strings are supported by the lexer if spaces are required. |

## Semantics
1. The config is first parsed by `NginxConfigParser`, producing an AST where each statement or block is a `NginxConfigStatement`.
2. A semantic pass walks the AST and builds a `ServerConfig` object:
   - Locate the top-level `server { ... }` block.
   - Extract the single port value from the `listen` directive inside that block.
   - For each `location` block, collect its `path` (from the directive), `handler` type, and arbitrary `options`.
   - Validate that paths begin with `/` and that no two locations share the same path.
3. `ServerConfig` is fed into the dispatcher factory:
   - The dispatcher sorts routes by descending prefix length so the first match equals the longest match.
   - Each route instantiates the requested handler (`EchoRequestHandler`, `StaticRequestHandler`, etc.) using the options map; unknown types should produce a descriptive error before the server binds.

## Sample Configs

Check `tests/integration/test_configs/` for runnable examples:

- `default_config` – minimal config wiring `/echo` to the echo handler and listening on `8080`.
- Future configs can define `/static` or other handler types; ensure every `location` block sets its `handler` and any required options.

When adding a new handler type:
1. Document its block options in this README.
2. Update the semantic parser to recognize the `type`.
3. Provide at least one sample config under `tests/integration/test_configs/` so integration tests can exercise it.
