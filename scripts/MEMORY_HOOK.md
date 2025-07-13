# Pre-Compact Memory Update Hook

## Overview

The Pre-Compact Memory Hook automatically preserves meaningful context from Claude Code sessions before the `/compact` command runs, preventing loss of important problem-solving information and technical decisions.

## Key Features

- 🔄 **Automatic activation** before every `/compact` command
- 🧠 **Intelligent context extraction** from session transcripts
- 💾 **MCP memory storage** of problems, solutions, and decisions
- 📈 **Relationship mapping** between entities
- 🔍 **Zero user intervention** required

## How It Works

1. **Triggers automatically** when `/compact` command is detected
2. **Analyzes session transcript** for meaningful content
3. **Extracts key information**:
   - Problems encountered
   - Solutions implemented
   - Code changes made
   - Technical decisions and rationale
4. **Creates MCP entities** and relationships
5. **Saves to memory database** for future reference

## What Gets Preserved

### 📋 Session Entity
Core session information with metrics:
```json
{
  "name": "Session_abc12345_20250707_193416",
  "entityType": "CodeSession", 
  "observations": [
    "Session ID: abc12345",
    "Timestamp: 20250707_193416",
    "Auto-saved before compact operation",
    "Problems addressed: 2",
    "Solutions implemented: 3",
    "Code changes made: 5"
  ]
}
```

### 🚨 Problem Entities
Issues, bugs, and challenges encountered:
```json
{
  "name": "Problem_abc12345_1",
  "entityType": "Problem",
  "observations": [
    "video streaming was very slow",
    "From session abc12345"
  ]
}
```

### ✅ Solution Entities  
Fixes and implementations applied:
```json
{
  "name": "Solution_abc12345_1", 
  "entityType": "Solution",
  "observations": [
    "implemented threading solution for video streaming",
    "From session abc12345"
  ]
}
```

### 🔧 Code Change Entities
Files modified during the session:
```json
{
  "name": "CodeChange_abc12345_1",
  "entityType": "CodeChange", 
  "observations": [
    "Modified /mnt/c/yard/app.py",
    "From session abc12345"
  ]
}
```

## Relationship Mapping

The hook creates meaningful relationships between entities:

- **Session** `encountered` **Problem**
- **Session** `implemented` **Solution** 
- **Session** `made_change` **CodeChange**
- **Solution** `solves` **Problem** (when detectable)

## Configuration

Located in `~/.claude/settings.json`:

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

## Context Extraction Patterns

### Problem Detection
Looks for patterns like:
- "error", "issue", "problem", "bug"
- "not working", "doesn't work", "fails to"
- "slow", "performance", "timeout"

### Solution Detection  
Identifies implementation language:
- "fixed", "solved", "resolved"
- "implemented", "updated", "changed"
- "created", "added", "improved"

### Code Change Detection
Tracks file modifications from:
- Edit tool calls
- Write tool calls  
- MultiEdit tool calls

## Example Memory Preservation

### Before Compact
Session working on video streaming performance:

**User Request**: "The video streaming is really slow, can you fix it?"

**Work Done**:
- Identified threading issue in `app.py`
- Implemented `CameraStream` class with threading
- Updated video capture to run at 30fps
- Fixed UI blocking problem

### After Hook Execution
**Entities Created**:
1. `Session_a1b2c3d4_20250707_143022` (CodeSession)
2. `Problem_a1b2c3d4_1` - "video streaming was really slow"
3. `Solution_a1b2c3d4_1` - "implemented threading solution"
4. `CodeChange_a1b2c3d4_1` - "Modified /mnt/c/yard/app.py"

**Relationships Created**:
- Session → Problem (encountered)
- Session → Solution (implemented)
- Session → CodeChange (made_change)

### Future Benefit
After `/compact`, when you ask "How did we fix the video streaming?":
- MCP memory search finds the session entities
- Returns the threading solution context
- Shows which files were modified
- Preserves the "why" behind the decision

## Monitoring

### Log File
Monitor hook activity:
```bash
tail -f /mnt/c/yard/scripts/hook.log
```

Example log output:
```
2025-07-07 19:34:16,260 - INFO - Pre-compact hook triggered for session abc123
2025-07-07 19:34:16,262 - INFO - Extracted context: 2 problems, 1 solutions, 1 code changes  
2025-07-07 19:34:16,305 - INFO - Successfully saved 5 entities and 4 relations to MCP memory
2025-07-07 19:34:16,306 - INFO - Successfully preserved session context before compact
```

### Memory File
Check what's being saved:
```bash
tail -10 /mnt/c/yard/mcp-memory/dist/memory.json
```

### MCP Memory Queries
After compacting, search preserved context:
```
search_nodes("video streaming threading")
search_nodes("performance issue solution")
open_nodes(["Session_abc12345_20250707_143022"])
```

## Testing

Run the test script to verify functionality:
```bash
python3 /mnt/c/yard/scripts/test_hook.py
```

Expected output:
```
Testing pre-compact memory update hook...
Hook exit code: 0
✓ Flask 2.3.3 ready
Successfully saved 5 entities and 4 relations to MCP memory
Successfully preserved session context before compact
Test PASSED
```

## Troubleshooting

### Hook Not Triggering
1. **Check hook configuration** in `~/.claude/settings.json`
2. **Verify script permissions**: `chmod +x pre_compact_memory_update.py`
3. **Check for Python errors** in hook.log

### No Context Extracted
- **Empty transcript**: Hook only saves meaningful content
- **Short sessions**: Minimum thresholds for problems/solutions
- **Check patterns**: May need to update extraction patterns

### Memory Not Updated  
1. **Verify MCP server running**: Check `/mcp` command
2. **Check memory file exists**: `/mnt/c/yard/mcp-memory/dist/memory.json`
3. **File permissions**: Ensure write access to memory file
4. **Review log file**: Check for specific error messages

### Script Timeout
- **Increase timeout** in settings.json (currently 30s)
- **Large transcripts** may need more processing time
- **Check system resources** if consistently timing out

## Benefits

### Context Continuity
- **Preserves "why" decisions** were made
- **Maintains solution history** across context resets
- **Searchable knowledge base** of past work

### Zero Effort
- **Completely automatic** - no manual intervention
- **Runs every time** before compact operations
- **Background operation** - doesn't interrupt workflow

### Rich Context
- **Problem-solution mapping** for future reference
- **Code change tracking** with file locations
- **Session-based organization** for chronological context

### Future Intelligence
- **Pattern recognition** across multiple sessions
- **Solution reuse** for similar problems
- **Project evolution tracking** over time

## Example Queries After Compacting

```bash
# Find all video-related problems
search_nodes("video streaming slow performance")

# Get specific session details  
open_nodes(["Session_abc12345_20250707_143022"])

# Find threading solutions
search_nodes("threading solution performance")

# See all code changes to app.py
search_nodes("Modified /mnt/c/yard/app.py")
```

The Memory Hook ensures that valuable problem-solving context is never lost, creating a persistent knowledge base that grows smarter with every session! 🧠💾