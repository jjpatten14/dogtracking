"""
GPU Video Acceleration Module

Provides GPU-accelerated video processing operations using CuPy and CUDA
to reduce video streaming latency from 2-4 seconds to 0.5-1 second.

Key Features:
- GPU frame resize and color conversion
- Memory-efficient GPU operations
- Async processing for reduced latency
- Fallback to CPU if GPU operations fail
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Union
from threading import Lock

# Configure logging
logger = logging.getLogger(__name__)

# GPU acceleration imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("✅ CuPy GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("⚠️ CuPy not available, using CPU fallback")

class GPUVideoProcessor:
    """GPU-accelerated video processing for real-time streaming"""
    
    def __init__(self, enable_gpu: bool = True):
        self.enable_gpu = enable_gpu and GPU_AVAILABLE
        self.gpu_memory_pool = None
        self.processing_lock = Lock()
        
        # Performance tracking
        self.stats = {
            'gpu_operations': 0,
            'cpu_fallbacks': 0,
            'avg_gpu_time': 0.0,
            'avg_cpu_time': 0.0,
            'total_frames_processed': 0
        }
        
        if self.enable_gpu:
            try:
                # Initialize GPU memory pool for efficient memory management
                self.gpu_memory_pool = cp.get_default_memory_pool()
                
                # Test GPU functionality
                self._test_gpu_operations()
                
                logger.info(f"🚀 GPU video processor initialized")
                logger.info(f"💾 GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
                
            except Exception as e:
                logger.error(f"GPU initialization failed: {e}")
                self.enable_gpu = False
        
        if not self.enable_gpu:
            logger.info("🔧 Using CPU-only video processing")
    
    def _test_gpu_operations(self):
        """Test basic GPU operations to ensure functionality"""
        try:
            # Test basic GPU array operations
            test_array = cp.random.randint(0, 255, (480, 640, 3), dtype=cp.uint8)
            resized = self._gpu_resize(test_array, (320, 240))
            converted = self._gpu_color_convert(resized, cv2.COLOR_BGR2RGB)
            
            # Test memory cleanup
            del test_array, resized, converted
            if self.gpu_memory_pool:
                self.gpu_memory_pool.free_all_blocks()
            
            logger.info("✅ GPU operations test passed")
            
        except Exception as e:
            logger.error(f"GPU operations test failed: {e}")
            raise
    
    def _gpu_resize(self, gpu_frame: cp.ndarray, target_size: Tuple[int, int]) -> cp.ndarray:
        """GPU-accelerated frame resize using CuPy"""
        try:
            h, w = target_size[1], target_size[0]  # (width, height) -> (height, width)
            
            # Use CuPy's interpolation for resizing
            # This is more efficient than transferring to CPU for cv2.resize
            original_h, original_w = gpu_frame.shape[:2]
            
            if original_h == h and original_w == w:
                return gpu_frame  # No resize needed
            
            # Calculate scaling factors
            scale_y = h / original_h
            scale_x = w / original_w
            
            # Create coordinate grids for interpolation
            y_coords = cp.arange(h, dtype=cp.float32) / scale_y
            x_coords = cp.arange(w, dtype=cp.float32) / scale_x
            
            # Use CuPy's map_coordinates for bilinear interpolation
            y_coords = cp.clip(y_coords, 0, original_h - 1)
            x_coords = cp.clip(x_coords, 0, original_w - 1)
            
            # Simple nearest neighbor for speed (can upgrade to bilinear if needed)
            y_indices = cp.round(y_coords).astype(cp.int32)
            x_indices = cp.round(x_coords).astype(cp.int32)
            
            # Create output frame
            if len(gpu_frame.shape) == 3:
                resized = gpu_frame[cp.ix_(y_indices, x_indices)]
            else:
                resized = gpu_frame[cp.ix_(y_indices, x_indices)]
            
            return resized
            
        except Exception as e:
            logger.error(f"GPU resize failed: {e}")
            raise
    
    def _gpu_color_convert(self, gpu_frame: cp.ndarray, conversion_code: int) -> cp.ndarray:
        """GPU-accelerated color conversion"""
        try:
            # For common conversions, implement directly in CuPy
            if conversion_code == cv2.COLOR_BGR2RGB:
                # Simple channel swap: BGR -> RGB
                return gpu_frame[:, :, [2, 1, 0]]
            elif conversion_code == cv2.COLOR_RGB2BGR:
                # Simple channel swap: RGB -> BGR  
                return gpu_frame[:, :, [2, 1, 0]]
            elif conversion_code == cv2.COLOR_BGR2GRAY:
                # Weighted average for grayscale
                weights = cp.array([0.114, 0.587, 0.299])  # BGR weights
                return cp.dot(gpu_frame, weights).astype(cp.uint8)
            else:
                # For other conversions, fallback to CPU
                cpu_frame = cp.asnumpy(gpu_frame)
                converted = cv2.cvtColor(cpu_frame, conversion_code)
                return cp.asarray(converted)
                
        except Exception as e:
            logger.error(f"GPU color conversion failed: {e}")
            raise
    
    def process_frame_gpu(self, frame: np.ndarray, 
                         target_size: Optional[Tuple[int, int]] = None,
                         color_conversion: Optional[int] = None) -> np.ndarray:
        """
        Process frame with GPU acceleration
        
        Args:
            frame: Input frame (numpy array)
            target_size: Target size (width, height) for resizing
            color_conversion: OpenCV color conversion code
            
        Returns:
            Processed frame (numpy array)
        """
        if not self.enable_gpu:
            return self._process_frame_cpu(frame, target_size, color_conversion)
        
        start_time = time.time()
        
        try:
            with self.processing_lock:
                # Transfer frame to GPU
                gpu_frame = cp.asarray(frame)
                
                # Apply GPU operations
                if target_size:
                    gpu_frame = self._gpu_resize(gpu_frame, target_size)
                
                if color_conversion:
                    gpu_frame = self._gpu_color_convert(gpu_frame, color_conversion)
                
                # Transfer back to CPU
                result = cp.asnumpy(gpu_frame)
                
                # Update stats
                processing_time = time.time() - start_time
                self.stats['gpu_operations'] += 1
                self.stats['total_frames_processed'] += 1
                
                # Update average GPU time
                gpu_ops = self.stats['gpu_operations']
                current_avg = self.stats['avg_gpu_time']
                self.stats['avg_gpu_time'] = (current_avg * (gpu_ops - 1) + processing_time) / gpu_ops
                
                logger.debug(f"GPU frame processing: {processing_time:.3f}s")
                
                return result
                
        except Exception as e:
            logger.warning(f"GPU processing failed, falling back to CPU: {e}")
            return self._process_frame_cpu(frame, target_size, color_conversion)
    
    def _process_frame_cpu(self, frame: np.ndarray,
                          target_size: Optional[Tuple[int, int]] = None,
                          color_conversion: Optional[int] = None) -> np.ndarray:
        """CPU fallback for frame processing"""
        start_time = time.time()
        
        result = frame.copy()
        
        if target_size:
            result = cv2.resize(result, target_size)
        
        if color_conversion:
            result = cv2.cvtColor(result, color_conversion)
        
        # Update stats
        processing_time = time.time() - start_time
        self.stats['cpu_fallbacks'] += 1
        self.stats['total_frames_processed'] += 1
        
        # Update average CPU time
        cpu_ops = self.stats['cpu_fallbacks']
        current_avg = self.stats['avg_cpu_time']
        self.stats['avg_cpu_time'] = (current_avg * (cpu_ops - 1) + processing_time) / cpu_ops
        
        logger.debug(f"CPU frame processing: {processing_time:.3f}s")
        
        return result
    
    def optimize_for_streaming(self, frame: np.ndarray, 
                             stream_resolution: Tuple[int, int] = (640, 480),
                             quality: int = 80) -> bytes:
        """
        Optimize frame for web streaming with minimal latency
        
        Args:
            frame: Input frame
            stream_resolution: Target streaming resolution (width, height)
            quality: JPEG quality (1-100)
            
        Returns:
            Encoded JPEG bytes
        """
        try:
            # GPU-accelerated resize for streaming
            processed_frame = self.process_frame_gpu(frame, target_size=stream_resolution)
            
            # Fast JPEG encoding
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, encoded = cv2.imencode('.jpg', processed_frame, encode_params)
            
            if success:
                return encoded.tobytes()
            else:
                logger.error("JPEG encoding failed")
                return b''
                
        except Exception as e:
            logger.error(f"Stream optimization failed: {e}")
            # Emergency fallback - basic encoding
            success, encoded = cv2.imencode('.jpg', frame)
            return encoded.tobytes() if success else b''
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        total_ops = self.stats['gpu_operations'] + self.stats['cpu_fallbacks']
        
        return {
            'gpu_available': self.enable_gpu,
            'gpu_operations': self.stats['gpu_operations'],
            'cpu_fallbacks': self.stats['cpu_fallbacks'],
            'gpu_usage_percent': (self.stats['gpu_operations'] / max(total_ops, 1)) * 100,
            'avg_gpu_time': self.stats['avg_gpu_time'],
            'avg_cpu_time': self.stats['avg_cpu_time'],
            'total_frames_processed': self.stats['total_frames_processed'],
            'gpu_memory_used': self._get_gpu_memory_usage()
        }
    
    def _get_gpu_memory_usage(self) -> dict:
        """Get GPU memory usage information"""
        if not self.enable_gpu:
            return {'available': False}
        
        try:
            memory_info = cp.cuda.runtime.memGetInfo()
            free_memory = memory_info[0]
            total_memory = memory_info[1]
            used_memory = total_memory - free_memory
            
            return {
                'available': True,
                'used_mb': used_memory / 1024 / 1024,
                'free_mb': free_memory / 1024 / 1024,
                'total_mb': total_memory / 1024 / 1024,
                'usage_percent': (used_memory / total_memory) * 100
            }
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {'available': False, 'error': str(e)}
    
    def cleanup(self):
        """Clean up GPU resources"""
        if self.enable_gpu and self.gpu_memory_pool:
            try:
                self.gpu_memory_pool.free_all_blocks()
                logger.info("🧹 GPU memory cleaned up")
            except Exception as e:
                logger.error(f"GPU cleanup error: {e}")

# Global GPU processor instance
_gpu_processor = None

def get_gpu_processor() -> GPUVideoProcessor:
    """Get global GPU processor instance"""
    global _gpu_processor
    if _gpu_processor is None:
        _gpu_processor = GPUVideoProcessor()
    return _gpu_processor

def is_gpu_available() -> bool:
    """Check if GPU acceleration is available"""
    return GPU_AVAILABLE

def process_frame_fast(frame: np.ndarray, 
                      target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Quick frame processing function"""
    processor = get_gpu_processor()
    return processor.process_frame_gpu(frame, target_size=target_size)

def encode_frame_fast(frame: np.ndarray, 
                     resolution: Tuple[int, int] = (640, 480),
                     quality: int = 80) -> bytes:
    """Quick frame encoding for streaming"""
    processor = get_gpu_processor()
    return processor.optimize_for_streaming(frame, resolution, quality)