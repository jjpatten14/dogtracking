"""
Central path management for Dog Tracking System
Provides portable paths that work on any OS, any drive, any folder location.
"""

from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """
    Detect the project root directory dynamically.
    
    Returns:
        Path: Absolute path to the project root directory
    """
    # Try environment variable override first
    if "DOG_TRACKING_HOME" in os.environ:
        custom_root = Path(os.environ["DOG_TRACKING_HOME"]).resolve()
        if custom_root.exists():
            logger.info(f"🏠 Using custom project root from DOG_TRACKING_HOME: {custom_root}")
            return custom_root
        else:
            logger.warning(f"⚠️ DOG_TRACKING_HOME path doesn't exist: {custom_root}")
    
    # Default: Use the directory containing this file as project root
    project_root = Path(__file__).parent.resolve()
    logger.info(f"🏠 Detected project root: {project_root}")
    return project_root

# Initialize project root
PROJECT_ROOT = get_project_root()

# Define all project paths relative to root
PROJECT_PATHS = {
    # Core directories
    'root': PROJECT_ROOT,
    'models': PROJECT_ROOT / 'models',
    'snapshots': PROJECT_ROOT / 'snapshots',
    'dogs': PROJECT_ROOT / 'dogs',
    'templates': PROJECT_ROOT / 'templates',
    'static': PROJECT_ROOT / 'static',
    'configs': PROJECT_ROOT / 'configs',
    'tts': PROJECT_ROOT / 'tts',
    'scripts': PROJECT_ROOT / 'scripts',
    
    # Subdirectories
    'dogs_annotations': PROJECT_ROOT / 'dogs' / 'annotations',
    'dogs_preprocessed': PROJECT_ROOT / 'dogs' / 'preprocessed',
    'dogs_profiles': PROJECT_ROOT / 'dogs' / 'profiles',
    'models_deployed': PROJECT_ROOT / 'models' / 'deployed',
    'models_templates': PROJECT_ROOT / 'models' / 'landmark_templates',
    'static_css': PROJECT_ROOT / 'static' / 'css',
    'static_js': PROJECT_ROOT / 'static' / 'js',
    'static_images': PROJECT_ROOT / 'static' / 'images',
    
    # Config files
    'boundary_config': PROJECT_ROOT / 'configs' / 'boundary_config.json',
    'alert_config': PROJECT_ROOT / 'configs' / 'alert_config.json',
    'camera_config': PROJECT_ROOT / 'configs' / 'camera_config.json',
    'camera_settings': PROJECT_ROOT / 'configs' / 'camera_settings.json',
    'reference_points': PROJECT_ROOT / 'configs' / 'reference_points.json',
    
    # Model files (with fallbacks)
    'megadetector_model': PROJECT_ROOT / 'models' / 'md_v5a.0.0.pt',  # Optional - fallback only
    'miewid_model': PROJECT_ROOT / 'models' / 'deployed' / 'latest_model.pth',
    'miewid_config': PROJECT_ROOT / 'models' / 'wbia-plugin-miew-id' / 'wbia_miew_id' / 'configs' / 'yard_dogs_config.yaml',
}

def get_path(key: str, create_parents: bool = True) -> Path:
    """
    Get a project path by key, optionally creating parent directories.
    
    Args:
        key: Path key from PROJECT_PATHS
        create_parents: Whether to create parent directories if they don't exist
        
    Returns:
        Path: Absolute path object
        
    Raises:
        KeyError: If the path key doesn't exist
    """
    if key not in PROJECT_PATHS:
        available_keys = ', '.join(PROJECT_PATHS.keys())
        raise KeyError(f"Unknown path key '{key}'. Available keys: {available_keys}")
    
    path = PROJECT_PATHS[key]
    
    if create_parents:
        # For files, create parent directory; for directories, create the directory itself
        if path.suffix:  # Has file extension, so it's a file
            path.parent.mkdir(parents=True, exist_ok=True)
        else:  # It's a directory
            path.mkdir(parents=True, exist_ok=True)
    
    return path

def get_snapshot_path(date_str: str = None, create_parents: bool = True) -> Path:
    """
    Get the snapshots directory for a specific date.
    
    Args:
        date_str: Date string (YYYY-MM-DD). If None, uses current date.
        create_parents: Whether to create the directory if it doesn't exist
        
    Returns:
        Path: Snapshots directory path for the date
    """
    if date_str is None:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    snapshot_path = get_path('snapshots') / date_str
    
    if create_parents:
        snapshot_path.mkdir(parents=True, exist_ok=True)
    
    return snapshot_path

def get_camera_template_path(camera_id: int, create_parents: bool = True) -> Path:
    """
    Get the landmark templates directory for a specific camera.
    
    Args:
        camera_id: Camera ID number
        create_parents: Whether to create the directory if it doesn't exist
        
    Returns:
        Path: Camera-specific template directory
    """
    template_path = get_path('models_templates') / f'camera_{camera_id}'
    
    if create_parents:
        template_path.mkdir(parents=True, exist_ok=True)
    
    return template_path

def migrate_legacy_configs():
    """
    Migrate config files from legacy hardcoded locations to new configs/ folder.
    This helps with backwards compatibility during the transition.
    """
    legacy_mappings = {
        # Old path -> New path key
        PROJECT_ROOT / 'boundary_config.json': 'boundary_config',
        PROJECT_ROOT / 'alert_config.json': 'alert_config', 
        PROJECT_ROOT / 'camera_config.json': 'camera_config',
        PROJECT_ROOT / 'camera_settings.json': 'camera_settings',
        PROJECT_ROOT / 'reference_points.json': 'reference_points',
    }
    
    migrated_files = []
    
    for old_path, new_key in legacy_mappings.items():
        if old_path.exists():
            new_path = get_path(new_key, create_parents=True)
            if not new_path.exists():
                # Copy old file to new location
                import shutil
                shutil.copy2(old_path, new_path)
                logger.info(f"📦 Migrated config: {old_path} -> {new_path}")
                migrated_files.append(str(old_path))
            else:
                logger.info(f"✅ Config already exists in new location: {new_path}")
    
    if migrated_files:
        logger.info(f"🔄 Migration complete. Migrated {len(migrated_files)} config files.")
        logger.info("💡 Old config files can be safely deleted after verifying the system works.")
    
    return migrated_files

def validate_project_structure():
    """
    Validate that the project structure is properly set up.
    Creates missing directories and reports any issues.
    
    Returns:
        bool: True if validation passed, False if there were issues
    """
    logger.info("🔍 Validating project structure...")
    
    issues = []
    
    # Check that we can write to the project root
    try:
        test_file = PROJECT_ROOT / '.write_test'
        test_file.write_text('test')
        test_file.unlink()
    except Exception as e:
        issues.append(f"Cannot write to project root: {e}")
    
    # Ensure all critical directories exist
    critical_dirs = ['models', 'snapshots', 'dogs', 'templates', 'static', 'configs']
    for dir_key in critical_dirs:
        try:
            get_path(dir_key, create_parents=True)
            logger.info(f"✅ Directory exists: {get_path(dir_key, create_parents=False)}")
        except Exception as e:
            issues.append(f"Cannot create directory '{dir_key}': {e}")
    
    # Check for optional model paths (don't create these)
    model_file = get_path('megadetector_model', create_parents=False)
    if not model_file.exists():
        logger.info(f"ℹ️ Local MegaDetector model not found: {model_file}")
        logger.info("💡 System will use MegaDetector package built-in model if available")
    
    if issues:
        logger.error("❌ Project structure validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("✅ Project structure validation passed!")
    return True

# Auto-migrate legacy configs when module is imported
if __name__ != "__main__":
    try:
        migrate_legacy_configs()
    except Exception as e:
        logger.warning(f"⚠️ Config migration failed: {e}")

# Convenience exports
__all__ = [
    'PROJECT_ROOT',
    'PROJECT_PATHS', 
    'get_path',
    'get_snapshot_path',
    'get_camera_template_path',
    'migrate_legacy_configs',
    'validate_project_structure'
]