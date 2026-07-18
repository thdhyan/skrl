# SKRL Project - Development Environment

## Conda Environment

**Active environment:** `belief`

**Path:** `/home/thakk100/miniconda3/envs/belief`

### Key Packages (Installed)

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10.20 | - |
| PyTorch | 2.10.0+cu126 | CUDA 12.6 enabled |
| Gymnasium | 1.3.0 | Environment interfaces |
| MuJoCo | 3.9.0 | Physics simulation |
| skrl | 2.1.0 (editable) | `/home/thakk100/Projects/skrl` |

### How to Activate

```bash
conda activate belief
cd /home/thakk100/Projects/skrl
```

### Running Tests

When running pytest or other Python commands, always use the activated `belief` environment:

```bash
conda activate belief
cd /home/thakk100/Projects/skrl
pytest tests/          # Run all tests
pytest tests/belief/   # Run belief-specific tests
```

## skills

This project uses the following Claude skills:

| Skill | Command | Purpose |
|-------|---------|---------|
| graphify | `/graphify` | Knowledge graph generation |

See `~/.claude/skills/graphify/SKILL.md` for skill documentation.

## Development Notes

- The project is installed in editable mode (`pip install -e .`)
- Changes to the codebase are immediately reflected without reinstallation
- CUDA-enabled PyTorch is available for GPU acceleration
- MuJoCo 3.9.0 is licensed software; ensure proper licensing for your use case

## Related Branches

- `feature/belief-pomdp-manipulation` — Current work on belief-based POMDP manipulation agent
- `develop` — Main development branch (for PRs)
- `main` — Release branch
