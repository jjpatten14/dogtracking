# MCP Database Setup Guide - Problem-Solution Context

## The Context Loss Problem

**Issue**: After code updates, when asking Claude to check memory, it can't find critical problem-solving context like "how we fixed the slow video streaming through threading."

**Root Cause**: MCP memory typically stores what exists, not why decisions were made or how problems were solved.

## Recommended MCP Entity Structure

### 1. Problem-Solution Pairs
Always create linked entities for issues and their solutions:

```json
{
  "name": "VideoStreamingPerformanceIssue",
  "entityType": "Problem",
  "observations": [
    "Web streaming was very slow initially",
    "Issue was difficult to diagnose and fix", 
    "Root cause was blocking video capture in main thread",
    "Symptoms: UI freezing, low frame rates, browser timeouts"
  ]
}

{
  "name": "ThreadingVideoSolution",
  "entityType": "Solution", 
  "observations": [
    "Implemented separate thread for camera capture in CameraStream class",
    "Frame updates run at ~30fps independent of web requests",
    "Prevents UI blocking during video processing",
    "Code location: app.py CameraStream.update() method lines 25-35",
    "Uses daemon thread with self.running flag for cleanup"
  ]
}
```

### 2. Technical Decision Context
Document why specific approaches were chosen:

```json
{
  "name": "ThreadingArchitectureDecision",
  "entityType": "TechnicalDecision",
  "observations": [
    "Chose threading over async/await for video capture",
    "Reason: OpenCV cv2.VideoCapture is blocking by design",
    "Alternative async approaches tested but failed",
    "Threading isolates video I/O from web serving",
    "Performance improvement: 5fps -> 30fps"
  ]
}
```

### 3. Code Location Mapping
Link problems to specific code locations:

```json
{
  "name": "CameraStreamClass",
  "entityType": "CodeComponent",
  "observations": [
    "Location: app.py lines 15-45",
    "Purpose: Threaded video capture and streaming",
    "Key methods: start(), update(), get_frame(), stop()",
    "Threading implementation solves performance issues",
    "Critical for video streaming functionality"
  ]
}
```

### 4. Performance Metrics
Document before/after measurements:

```json
{
  "name": "VideoPerformanceMetrics", 
  "entityType": "PerformanceData",
  "observations": [
    "Before threading: 3-5 fps, UI freezing",
    "After threading: 30 fps, responsive UI", 
    "Memory usage: stable ~50MB",
    "CPU usage: 15-20% single core",
    "Browser compatibility: Chrome, Firefox, Edge tested"
  ]
}
```

## Relationship Patterns

### Problem -> Solution -> Implementation
```
VideoStreamingPerformanceIssue --solved_by--> ThreadingVideoSolution
ThreadingVideoSolution --implemented_in--> CameraStreamClass  
ThreadingVideoSolution --measured_by--> VideoPerformanceMetrics
```

### Code -> Purpose -> Context
```
CameraStreamClass --solves--> VideoStreamingPerformanceIssue
CameraStreamClass --uses_pattern--> ThreadingArchitectureDecision
```

## MCP Setup Commands

### 1. Create Problem-Solution Entities
```bash
# After deploying MCP, run these commands in Claude Code:

# Document the video streaming issue
create_entities([{
  "name": "VideoStreamingPerformanceIssue",
  "entityType": "Problem", 
  "observations": ["Web streaming was very slow initially", "Required threading solution"]
}])

# Document the solution
create_entities([{
  "name": "ThreadingVideoSolution",
  "entityType": "Solution",
  "observations": ["Separate thread for camera capture", "Located in app.py CameraStream class"]
}])

# Link them
create_relations([{
  "from": "VideoStreamingPerformanceIssue",
  "to": "ThreadingVideoSolution", 
  "relationType": "solved_by"
}])
```

### 2. Map Code Components
```bash
# Document where the solution lives
create_entities([{
  "name": "CameraStreamClass",
  "entityType": "CodeComponent",
  "observations": ["app.py lines 15-45", "Threaded video capture implementation"]
}])

# Link solution to implementation  
create_relations([{
  "from": "ThreadingVideoSolution",
  "to": "CameraStreamClass",
  "relationType": "implemented_in"
}])
```

### 3. Add Performance Context
```bash
create_entities([{
  "name": "VideoPerformanceMetrics",
  "entityType": "PerformanceData", 
  "observations": ["Before: 3-5 fps", "After: 30 fps", "Threading solved UI blocking"]
}])
```

## Search Examples

After proper setup, these searches should work:

```bash
# Find performance issues
search_nodes("slow video")
search_nodes("streaming performance") 
search_nodes("threading")

# Find solutions to specific problems
search_nodes("video streaming fix")
search_nodes("UI blocking solution")

# Find code related to performance
search_nodes("CameraStream threading")
```

## Maintenance Guidelines

### 1. Document Every Major Fix
- Create Problem entity when issue discovered
- Create Solution entity when fix implemented  
- Link them with "solved_by" relationship
- Add code location and performance impact

### 2. Update When Code Changes
- Add observations when modifying solutions
- Link new components that use patterns
- Update performance metrics if measured

### 3. Use Consistent Entity Types
- **Problem**: Issues, bugs, performance problems
- **Solution**: Fixes, implementations, workarounds  
- **TechnicalDecision**: Architecture choices, trade-offs
- **CodeComponent**: Classes, functions, modules
- **PerformanceData**: Before/after metrics, benchmarks

## Template for Future Issues

When solving any significant problem:

1. **Create Problem Entity**: Document symptoms, root cause, difficulty level
2. **Create Solution Entity**: Document approach, implementation details, code location
3. **Create Decision Entity**: Document why this approach over alternatives
4. **Create Performance Entity**: Document measurable improvements
5. **Link Everything**: Create relationships between all entities

This ensures future Claude instances can understand not just what exists, but why it exists and how problems were solved.