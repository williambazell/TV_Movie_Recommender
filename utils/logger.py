import logging
import json
import time
import threading
import sys
from typing import Dict, Any
from datetime import datetime
from functools import wraps

class ProgressIndicator:
    def __init__(self, message: str = "Loading", verbose: bool = False):
        self.message = message
        self.verbose = verbose
        self.running = False
        self.thread = None
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.current_char = 0
        
    def start(self):
        if not self.verbose:
            self.running = True
            self.thread = threading.Thread(target=self._animate)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self, final_message: str = None):
        self.running = False
        if self.thread:
            self.thread.join()

        if not self.verbose and final_message:
            sys.stdout.write(f'\r{final_message}\n')
            sys.stdout.flush()
    
    def _animate(self):
        while self.running:
            char = self.spinner_chars[self.current_char]
            sys.stdout.write(f'\r{self.message} {char}')
            sys.stdout.flush()
            self.current_char = (self.current_char + 1) % len(self.spinner_chars)
            time.sleep(0.1)
    
    def update_message(self, new_message: str):
        self.message = new_message

class CondensedLogger:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.progress_indicators = {}
        self.operation_groups = {}
        
    def start_operation_group(self, group_name: str, message: str = None):
        if not self.verbose:
            if message is None:
                message = f"Loading {group_name.replace('_', ' ').title()}"
            self.operation_groups[group_name] = ProgressIndicator(message, self.verbose)
            self.operation_groups[group_name].start()
        else:
            print(f"Starting {group_name}...")
    
    def end_operation_group(self, group_name: str, message: str = None):
        if not self.verbose:
            if group_name in self.operation_groups:
                if message is None:
                    message = f"✓ {group_name.replace('_', ' ').title()} completed"
                self.operation_groups[group_name].stop(message)
                del self.operation_groups[group_name]
        else:
            if message is None:
                message = f"{group_name.replace('_', ' ').title()} completed"
            print(message)
    
    def update_operation_group(self, group_name: str, message: str):
        if not self.verbose and group_name in self.operation_groups:
            self.operation_groups[group_name].update_message(message)
    
    def start_operation(self, operation: str, message: str = None):
        if not self.verbose:
            if message is None:
                message = f"Loading {operation.replace('_', ' ').title()}"
            self.progress_indicators[operation] = ProgressIndicator(message, self.verbose)
            self.progress_indicators[operation].start()
        else:
            print(f"Starting {operation}...")
    
    def end_operation(self, operation: str, message: str = None):
        if not self.verbose:
            if operation in self.progress_indicators:
                if message is None:
                    message = f"✓ {operation.replace('_', ' ').title()} completed"
                self.progress_indicators[operation].stop(message)
                del self.progress_indicators[operation]
        else:
            if message is None:
                message = f"{operation.replace('_', ' ').title()} completed"
            print(message)
    
    def update_operation(self, operation: str, message: str):
        if not self.verbose and operation in self.progress_indicators:
            self.progress_indicators[operation].update_message(message)

class MetricsCollector:
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str):
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        if operation not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        if 'timings' not in self.metrics:
            self.metrics['timings'] = {}
        
        if operation not in self.metrics['timings']:
            self.metrics['timings'][operation] = []
        
        self.metrics['timings'][operation].append(duration)
        return duration
    
    def increment_counter(self, counter: str, value: int = 1):
        if 'counters' not in self.metrics:
            self.metrics['counters'] = {}
        
        self.metrics['counters'][counter] = self.metrics['counters'].get(counter, 0) + value
    
    def set_gauge(self, gauge: str, value: float):
        if 'gauges' not in self.metrics:
            self.metrics['gauges'] = {}
        
        self.metrics['gauges'][gauge] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()
    
    def clear_metrics(self):
        self.metrics.clear()
        self.start_times.clear()

metrics = MetricsCollector()

class StructuredLogger:
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler('recommendations.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_structured(self, level: str, message: str, **kwargs):
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        
        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_data, default=str))
    
    def info(self, message: str, **kwargs):
        self.log_structured('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log_structured('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log_structured('ERROR', message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self.log_structured('DEBUG', message, **kwargs)

def log_performance(operation: str, verbose: bool = False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics.start_timer(operation)
            try:
                result = func(*args, **kwargs)
                metrics.increment_counter(f"{operation}_success")
                return result
            except Exception as e:
                metrics.increment_counter(f"{operation}_error")
                raise
            finally:
                duration = metrics.end_timer(operation)
                if verbose or (verbose_api_logging and 'api_call' in operation):
                    logger.info(f"Operation {operation} completed", 
                               operation=operation, 
                               duration=duration)
        return wrapper
    return decorator

def log_model_metrics(model_name: str, metrics_data: Dict[str, Any]):
    logger.info("Model metrics", model_name=model_name, metrics=metrics_data)
    
logger = StructuredLogger("recommendation_system")

def monitor_api_call(func):
    return log_performance(f"api_call_{func.__name__}", verbose=False)(func)

def monitor_model_inference(func):
    return log_performance(f"model_inference_{func.__name__}", verbose=False)(func)

def monitor_data_processing(func):
    return log_performance(f"data_processing_{func.__name__}", verbose=False)(func)

condensed_logger = CondensedLogger(verbose=False)

verbose_api_logging = False

def enable_verbose_api_logging():
    global verbose_api_logging
    verbose_api_logging = True

def disable_verbose_api_logging():
    global verbose_api_logging
    verbose_api_logging = False
