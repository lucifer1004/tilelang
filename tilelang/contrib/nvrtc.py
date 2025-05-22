import logging
import multiprocessing as mp
from typing import List, Literal, Optional, Tuple, Union
import threading
import atexit
import os # For debugging in worker

import cuda.bindings.nvrtc as nvrtc
from tvm.target import Target

from .nvcc import get_target_compute_version

logger = logging.getLogger(__name__)

# Define module-level constants and state for managing compiler subprocesses
_active_compiler_processes = []
_compiler_processes_lock = threading.Lock()
_thread_local_compiler_data = threading.local()


def _cleanup_compiler_processes():
    """Gracefully shut down all active compiler subprocesses."""
    with _compiler_processes_lock:
        # Create a copy for iteration as _active_compiler_processes might be modified elsewhere
        # if not careful, though cleanup should be the last thing.
        processes_to_clean = list(_active_compiler_processes) # Now contains (req_q, res_q, proc)
        _active_compiler_processes.clear() # Clear original list

    for req_q, res_q, proc in processes_to_clean:
        try:
            logger.debug(f"Sending exit signal to compiler worker {proc.pid}")
            req_q.put(None) # Signal worker to exit
        except Exception as e:
            # Queue might be closed or process already terminated
            logger.error(f"Error sending exit signal to {proc.pid} via req_q: {e}")
            pass # Continue to attempt cleanup

    for req_q, res_q, proc in processes_to_clean:
        try:
            logger.debug(f"Closing queues for compiler worker {proc.pid}")
            req_q.close()
            req_q.join_thread() # Wait for the queue's feeder thread
        except Exception as e:
            logger.error(f"Error closing request queue for {proc.pid}: {e}")
        try:
            res_q.close()
            res_q.join_thread() # Wait for the queue's feeder thread
        except Exception as e:
            logger.error(f"Error closing result queue for {proc.pid}: {e}")

        try:
            logger.debug(f"Joining compiler worker {proc.pid}")
            proc.join(timeout=5)  # Wait for 5 seconds
            if proc.is_alive():
                logger.warning(f"Compiler worker {proc.pid} did not exit gracefully, terminating.")
                proc.terminate()
                proc.join(timeout=2) # Wait for termination
        except Exception as e:
            logger.error(f"Error joining/terminating process {proc.pid}: {e}")
            pass

atexit.register(_cleanup_compiler_processes)


def _compile_cuda_worker_process(request_queue: mp.Queue, result_queue: mp.Queue):
    """Worker function running in a separate process to handle CUDA compilation requests."""
    # This function runs in the spawned process.
    logger.debug(f"Compiler worker process {os.getpid()} started.")
    while True:
        try:
            task = request_queue.get()
            if task is None:
                logger.debug(f"Compiler worker process {os.getpid()} received exit signal.")
                break
            
            code, target_format, arch, options, verbose_in_task = task
            if verbose_in_task:
                logger.info(f"Compiler worker {os.getpid()} received task.")
            compiled_code = _compile_cuda(code, target_format, arch, options, verbose_in_task)
            result_queue.put(compiled_code)
            if verbose_in_task:
                logger.info(f"Compiler worker {os.getpid()} completed task.")
        except Exception as e:
            logger.error(f"Compiler worker {os.getpid()} encountered error: {e}", exc_info=True)
            result_queue.put(e)
    logger.debug(f"Compiler worker process {os.getpid()} exiting.")


def get_nvrtc_version() -> Tuple[int, int]:
    result, major, minor = nvrtc.nvrtcVersion()
    assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get NVRTC version: {result}"
    return (major, minor)


def _compile_cuda(code: str,
                 target_format: Literal["ptx", "cubin"] = "ptx",
                 arch: Optional[int] = None,
                 options: Optional[Union[str, List[str]]] = None,
                 verbose: bool = False) -> bytearray:
    """Compile cuda code with NVRTC.

    Parameters
    ----------
    code : str
        The cuda code.

    target_format : Literal["ptx", "cubin"]
        The target format of nvrtc compiler.

    arch : Optional[int]
        The cuda architecture code.

    options : Optional[Union[str, List[str]]]
        The additional options.

    verbose : bool
        Whether to print the verbose output.

    Return
    ------
    result_bytes : bytearray
        The bytearray of the cubin or ptx code.
    """
    if arch is None:
        # If None, then it will use `tvm.target.Target.current().arch`.
        # Target arch could be a str like "80", "90", "90a", etc.
        compute_version = "".join(
            get_target_compute_version(Target.current(allow_none=True)).split("."))
        arch = int(compute_version)
    prefix = "compute" if target_format == "ptx" else "sm"
    suffix = "a" if arch >= 90 else ""
    arch_option = f"--gpu-architecture={prefix}_{arch}{suffix}"

    file_name = "tvm_kernels"
    if target_format not in ["cubin", "ptx"]:
        raise ValueError("target_format must be cubin or ptx")

    final_options = ["-default-device", "-std=c++17"]
    if get_nvrtc_version() >= (12, 8):
        final_options += ["-pch"]
    if arch is not None:
        final_options += [arch_option]

    if options:
        if isinstance(options, str):
            final_options += [options]
        elif isinstance(options, list):
            final_options += options
        else:
            raise ValueError("options must be str or list of str")

    code = "#include <tl_templates/cuda/nvrtc_std.h>\n" + code
    code_bytes = bytes(code, "utf-8")
    result, program = nvrtc.nvrtcCreateProgram(
        code_bytes, bytes(file_name, "utf-8"), 0, [], [])
    assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to create program: {result}"

    options_bytes = [bytes(flag, "utf-8") for flag in final_options]
    compile_result = nvrtc.nvrtcCompileProgram(
        program, len(options_bytes), options_bytes)[0]

    if compile_result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        msg = f"{code}\n" \
            f"Compilation error:\n"
        if verbose:
            result, log_size = nvrtc.nvrtcGetProgramLogSize(program)
            assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get program log size: {result}"
            log_bytes = bytes(log_size)
            result = nvrtc.nvrtcGetProgramLog(program, log_bytes)[0]
            assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get program log: {result}"
            msg += f"{log_bytes.decode('utf-8')}\n"
        else:
            msg += "Turn on verbose to see the full compilation log."
        msg += f"Options: {' '.join(final_options)}\n"
        raise RuntimeError(msg)

    if target_format == "cubin":
        result, cubin_size = nvrtc.nvrtcGetCUBINSize(program)
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get CUBIN size: {result}"
        result_bytes = bytes(cubin_size)
        result = nvrtc.nvrtcGetCUBIN(program, result_bytes)[0]
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get CUBIN: {result}"
    else:
        result, ptx_size = nvrtc.nvrtcGetPTXSize(program)
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get PTX size: {result}"
        result_bytes = bytes(ptx_size)
        result = nvrtc.nvrtcGetPTX(program, result_bytes)[0]
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get PTX: {result}"

    # Destroy handler
    assert nvrtc.nvrtcDestroyProgram(
        program)[0] == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to destroy program: {result}"

    return result_bytes


def compile_cuda(code: str,
                 target_format: Literal["ptx", "cubin"] = "ptx",
                 arch: Optional[int] = None,
                 options: Optional[Union[str, List[str]]] = None,
                 verbose: bool = False,
                 subprocess: bool = False) -> bytearray:
    if subprocess:
        # Initialize thread-local compiler process if it doesn't exist for this thread
        if not hasattr(_thread_local_compiler_data, 'request_queue'):
            if verbose:
                logger.info(f"Main thread {threading.get_ident()}: Creating new compiler process.")
            ctx = mp.get_context('spawn')
            req_q = ctx.Queue()
            res_q = ctx.Queue()
            
            # _compile_cuda_worker_process needs access to _compile_cuda
            # which are defined in this module.
            # Correctly assign to proc before using it
            process_handle_for_thread = ctx.Process(target=_compile_cuda_worker_process, args=(req_q, res_q))
            # proc.daemon = True # Rely on atexit for cleaner shutdown
            process_handle_for_thread.start()
            if verbose:
                logger.info(f"Main thread {threading.get_ident()}: Started compiler process {process_handle_for_thread.pid}.")

            _thread_local_compiler_data.request_queue = req_q
            _thread_local_compiler_data.result_queue = res_q
            _thread_local_compiler_data.process = process_handle_for_thread # Store the correct process handle

            with _compiler_processes_lock:
                _active_compiler_processes.append((req_q, res_q, process_handle_for_thread)) # Store req_q, res_q, and process
        
        request_queue = _thread_local_compiler_data.request_queue
        result_queue = _thread_local_compiler_data.result_queue
        process_handle = _thread_local_compiler_data.process

        if not process_handle.is_alive():
            # This case should ideally be handled by recreating the process,
            # or by the user ensuring threads don't outlive main program cleanup.
            # For now, raise an error if the dedicated process died unexpectedly.
            logger.error(f"Error: Compiler process for thread {threading.get_ident()} (PID: {process_handle.pid}) is not alive.")
            # Attempt to clean up this thread's specific entry and raise
            try:
                with _compiler_processes_lock:
                    # Find and remove the specific dead process entry if it's still there
                    entry_to_remove_tuple = None
                    found_idx = -1
                    for i, (entry_req_q, entry_res_q, entry_proc) in enumerate(_active_compiler_processes):
                        if entry_proc is process_handle: # Check if it's the same process object
                            found_idx = i
                            entry_to_remove_tuple = (entry_req_q, entry_res_q, entry_proc)
                            break
                    if found_idx != -1:
                        del _active_compiler_processes[found_idx]
                        logger.debug(f"Removed dead process entry for PID {process_handle.pid} from _active_compiler_processes.")
                        
                        # Explicitly close queues of the dead process
                        if entry_to_remove_tuple:
                            dead_req_q, dead_res_q, _ = entry_to_remove_tuple
                            try:
                                dead_req_q.close()
                                dead_req_q.join_thread()
                            except Exception as e_rq_close:
                                logger.error(f"Error closing request queue for dead PID {process_handle.pid}: {e_rq_close}")
                            try:
                                dead_res_q.close()
                                dead_res_q.join_thread()
                            except Exception as e_rsq_close:
                                logger.error(f"Error closing result queue for dead PID {process_handle.pid}: {e_rsq_close}")

            except Exception as e_cleanup:
                logger.error(f"Error during cleanup of dead process entry for PID {process_handle.pid}: {e_cleanup}", exc_info=True)
            # Clear this thread's local data to force re-creation on next call
            # Do this after attempting to close queues associated with the _thread_local_compiler_data
            local_req_q_to_clean = getattr(_thread_local_compiler_data, 'request_queue', None)
            local_res_q_to_clean = getattr(_thread_local_compiler_data, 'result_queue', None)

            if local_req_q_to_clean:
                try:
                    local_req_q_to_clean.close()
                    local_req_q_to_clean.join_thread()
                except Exception as e:
                    logger.error(f"Error cleaning up thread-local request_queue for dead PID {process_handle.pid}: {e}")
            if local_res_q_to_clean:
                try:
                    local_res_q_to_clean.close()
                    local_res_q_to_clean.join_thread()
                except Exception as e:
                    logger.error(f"Error cleaning up thread-local result_queue for dead PID {process_handle.pid}: {e}")

            if hasattr(_thread_local_compiler_data, 'request_queue'): del _thread_local_compiler_data.request_queue
            if hasattr(_thread_local_compiler_data, 'result_queue'): del _thread_local_compiler_data.result_queue
            if hasattr(_thread_local_compiler_data, 'process'): del _thread_local_compiler_data.process
            raise RuntimeError(f"Compiler subprocess for thread {threading.get_ident()} (PID: {process_handle.pid}) died unexpectedly.")

        # Send compilation task
        task_payload = (code, target_format, arch, options, verbose)
        if verbose:
            logger.info(f"Main thread {threading.get_ident()}: Sending task to compiler process {process_handle.pid}.")
        request_queue.put(task_payload)
        
        # Retrieve result
        result = result_queue.get()
        if verbose:
            logger.info(f"Main thread {threading.get_ident()}: Received result from compiler process {process_handle.pid}.")
        
        if isinstance(result, Exception):
            # Consider logging the error source more explicitly
            logger.error(f"Error received from compiler subprocess {process_handle.pid}: {type(result).__name__} - {result}", exc_info=isinstance(result, BaseException))
            raise result
        return result
    
    # Original non-subprocess path
    return _compile_cuda(code, target_format, arch, options, verbose)