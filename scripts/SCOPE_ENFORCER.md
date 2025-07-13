# Claude Code Scope Enforcer Hook

## Overview

The Scope Enforcer is a PreToolUse hook that prevents Claude from going off-rails and adding unnecessary features, documentation, or complexity that wasn't requested in the original task.

## Key Features

- 🎯 **Keeps Claude focused** on the original user request
- 🚫 **Blocks scope creep** and unwanted feature additions  
- 🔓 **Easy disable mechanism** with magic keywords
- 📝 **Prevents unwanted documentation** creation
- ⚠️ **Blocks dangerous commands**
- 📖 **Allows information gathering** (Read operations)

## How It Works

1. **Intercepts every tool call** before execution
2. **Analyzes relevance** to the original user request
3. **Detects scope creep patterns** in Claude's responses
4. **Blocks or approves** actions with detailed reasoning

## Disable Keywords

Add any of these keywords to your message to bypass scope enforcement:

- `no rails` - Disables scope checking
- `off rails` - Same as above
- `free form` - Allows unrestricted actions  
- `no scope` - Bypasses scope enforcement

**Examples:**
```
no rails - help me explore this entire codebase and create comprehensive documentation

off rails - I want you to improve everything you see, add features, optimize performance

free form - refactor the whole project however you think is best
```

## What Gets Blocked

### 🚫 Unwanted Documentation
- Creating README files when not requested
- Adding documentation without explicit ask
- Generating guides, changelogs, etc.

### 🚫 Scope Creep Language
- "While we're at it..."
- "Let me also add..."
- "I'll also improve..."
- "We might as well..."

### 🚫 Dangerous Operations
- `rm -rf` commands
- Sudo operations
- System-level modifications

### 🚫 Feature Additions
- Adding functionality not in original request
- "Improving" code beyond stated problem
- Creating new components unprompted

## What Gets Approved

### ✅ Direct Task Solutions
- Fixing specifically mentioned issues
- Modifying explicitly requested files
- Implementing stated requirements

### ✅ Information Gathering
- Reading files to understand code
- Listing directory contents
- Searching for patterns

### ✅ Core Functionality
- Essential changes to solve the problem
- Direct responses to user requests

## Configuration

Located in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /mnt/c/yard/scripts/scope_enforcer.py",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

## Example Scenarios

### ❌ Blocked: Unwanted Documentation
**User:** "Fix the login bug"  
**Claude tries:** Create `/docs/LOGIN_SYSTEM.md`  
**Result:** BLOCKED - "Creating documentation file not requested"

### ❌ Blocked: Scope Creep  
**User:** "Change button color to blue"  
**Claude tries:** Refactor entire CSS system  
**Result:** BLOCKED - "Action not relevant to button color change"

### ✅ Approved: Direct Fix
**User:** "Fix the login bug"  
**Claude tries:** Edit `auth.py` to fix validation  
**Result:** APPROVED - "Action relevant to fixing login bug"

### ✅ Approved: Disabled Mode
**User:** "no rails - improve the entire UI system"  
**Claude tries:** Create new CSS framework  
**Result:** APPROVED - "Scope enforcement disabled by user"

## Logging

All scope enforcement decisions are logged to:
```
/mnt/c/yard/scripts/scope_enforcer.log
```

Monitor with:
```bash
tail -f /mnt/c/yard/scripts/scope_enforcer.log
```

## Testing

Run the test suite to verify functionality:
```bash
python3 /mnt/c/yard/scripts/test_scope_enforcer.py
```

## Troubleshooting

### Hook Not Working
1. Check settings.json configuration
2. Verify script permissions: `chmod +x scope_enforcer.py`
3. Check log file for errors

### Too Many Blocks
1. Add `no rails` to your message for creative freedom
2. Be more specific about what you want
3. Adjust patterns in the script if needed

### Not Enough Blocks
1. Review and enhance scope creep patterns
2. Add more file type restrictions
3. Lower relevance thresholds

## Benefits

- **Saves time** by preventing unnecessary work
- **Maintains focus** on actual user needs
- **Prevents over-engineering** and feature creep
- **User maintains control** with disable keywords
- **Improves efficiency** by blocking scope drift

The Scope Enforcer keeps Claude laser-focused on your actual requests while giving you full control when you need creative freedom! 🎯