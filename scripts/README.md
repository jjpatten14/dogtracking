# Pre-Compact Memory Update Hook

## Overview

This hook automatically preserves meaningful context from Claude Code sessions before the `/compact` command runs, preventing loss of important problem-solving information and technical decisions.

## How It Works

1. **Trigger**: Activates automatically before any `/compact` command
2. **Analysis**: Parses the session transcript to extract:
   - Problems encountered
   - Solutions implemented  
   - Code changes made
   - Technical decisions and rationale
3. **Storage**: Saves extracted context to MCP memory database
4. **Preservation**: Maintains "why" context that would otherwise be lost

## Files

- `pre_compact_memory_update.py` - Main hook script
- `test_hook.py` - Test script for verification
- `hook.log` - Log file for hook operations
- `README.md` - This documentation

## Configuration

The hook is configured in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "compact",
        "hooks": [
          {
            "type": "command", 
            "command": "python3 /mnt/c/yard/scripts/pre_compact_memory_update.py",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

## What Gets Preserved

### Entities Created:
- **CodeSession**: Overall session summary with metrics
- **Problem**: Issues/bugs/errors encountered
- **Solution**: Fixes and implementations applied
- **CodeChange**: Files modified during the session

### Relationships:
- Session → Problems (encountered)
- Session → Solutions (implemented)  
- Session → CodeChanges (made_change)

## Example Memory Entries

After a session where video streaming performance was fixed:

```json
{"type": "entity", "name": "Session_abc12345_20250707_193416", "entityType": "CodeSession", "observations": ["Problems addressed: 1", "Solutions implemented: 1"]}

{"type": "entity", "name": "Problem_abc12345_1", "entityType": "Problem", "observations": ["video streaming being very slow", "From session abc12345"]}

{"type": "entity", "name": "Solution_abc12345_1", "entityType": "Solution", "observations": ["implemented threading solution for video streaming", "From session abc12345"]}
```

## Testing

Run the test script to verify functionality:

```bash
cd /mnt/c/yard/scripts
python3 test_hook.py
```

## Monitoring

Check the log file for hook operations:

```bash
tail -f /mnt/c/yard/scripts/hook.log
```

## Troubleshooting

### Hook Not Running
- Verify hook configuration in `~/.claude/settings.json`
- Check Python script permissions: `chmod +x pre_compact_memory_update.py`
- Check log file for errors

### Memory Not Updated
- Verify MCP memory server is running
- Check memory file exists: `/mnt/c/yard/mcp-memory/dist/memory.json`
- Review hook.log for error messages

### Script Errors
- Ensure Python 3 is installed and accessible
- Check script syntax: `python3 -m py_compile pre_compact_memory_update.py`
- Verify file paths are correct

## Maintenance

### Regular Tasks:
1. **Monitor log file size** - rotate if it gets too large
2. **Check memory file growth** - archive old entries if needed
3. **Update extraction patterns** - add new problem/solution patterns as needed

### Upgrades:
- Update entity types as project evolves
- Add new relationship types
- Improve context extraction algorithms

## Benefits

- **Zero-effort preservation** of session context
- **Automatic before every compact** - no manual intervention
- **Preserves "why" decisions** - not just what was done
- **Maintains continuity** across context resets
- **Searchable history** via MCP memory queries

This hook ensures that valuable problem-solving context is never lost, even after aggressive context compacting.